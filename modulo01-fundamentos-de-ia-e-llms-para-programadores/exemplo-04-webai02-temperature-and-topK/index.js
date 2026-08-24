
const aiContext = {
    session: null,
    abortController: null,
    isGenerating: false,
};

const elements = {
    temperature: document.getElementById('temperature'),
    temperatureValue: document.getElementById('temp-value'),
    topKValue: document.getElementById('topk-value'),
    topK: document.getElementById('topK'),
    form: document.getElementById('question-form'),
    questionInput: document.getElementById('question'),
    output: document.getElementById('output'),
    button: document.getElementById('ask-button'),
    year: document.getElementById('year'),
}

function setupEventListeners() {

    // Update display values for range inputs
    elements.temperature.addEventListener('input', (e) => {
        elements.temperatureValue.textContent = e.target.value;
    });

    elements.topK.addEventListener('input', (e) => {
        elements.topKValue.textContent = e.target.value;
    });

    elements.form.addEventListener('submit', async function (event) {
        event.preventDefault();

        if (aiContext.isGenerating) {
            toggleSendOrStopButton(false)
            return;
        }

        onSubmitQuestion();
    });
}

async function onSubmitQuestion() {
    const questionInput = elements.questionInput;
    const output = elements.output;
    const question = questionInput.value;

    if (!question.trim()) {
        return;
    }

    // Get parameters from form
    const temperature = parseFloat(elements.temperature.value);
    const topK = parseInt(elements.topK.value);
    console.log('Using parameters:', { temperature, topK });

    // Change button to stop mode
    toggleSendOrStopButton(true)

    output.textContent = 'Processing your question...';

    try {
        const aiResponseChunks = askAI(question, temperature, topK);
        output.textContent = '';

        for await (const chunk of aiResponseChunks) {
            if (aiContext.abortController.signal.aborted) {
                break;
            }
            console.log('Received chunk:', chunk);
            output.textContent += chunk;
        }
    } catch (error) {
        if (error.name === 'AbortError') {
            console.log('Generation aborted by the user');
        } else {
            console.error('Error while prompting the model:', error);
            output.textContent = `⚠️ Erro ao conversar com o modelo: ${error.message}`;
        }
    }

    toggleSendOrStopButton(false);
}

function toggleSendOrStopButton(isGenerating) {
    if (isGenerating) {
        // Switch to stop mode
        aiContext.isGenerating = isGenerating;
        elements.button.textContent = 'Parar';
        elements.button.classList.add('stop-button');
    } else {
        // Switch to send mode
        aiContext.abortController?.abort();
        aiContext.isGenerating = isGenerating;
        elements.button.textContent = 'Enviar';
        elements.button.classList.remove('stop-button');
    }
}

async function* askAI(question, temperature, topK) {
    aiContext.abortController?.abort();
    aiContext.abortController = new AbortController();

    // Destroy previous session and create new one with updated parameters
    if (aiContext.session) {
        aiContext.session.destroy();
        aiContext.session = null;
    }

    const session = await LanguageModel.create({
        expectedInputLanguages: ["pt"],
        temperature: temperature,
        topK: topK,
        initialPrompts: [
            {
                role: 'system', content: `
                Você é um assistente de IA que responde de forma clara e objetiva.
                Responda sempre em formato de texto ao invés de markdown`

            },
        ],
    });
    aiContext.session = session;

    const responseStream = await session.promptStreaming(
        [
            {
                role: 'user',
                content: question,
            },
        ],
        {
            signal: aiContext.abortController.signal,
        }
    );

    for await (const chunk of responseStream) {
        if (aiContext.abortController.signal.aborted) {
            break;
        }
        yield chunk;
    }
}

/**
 * Verifica o ambiente SEM tentar baixar o modelo.
 * O download precisa de um gesto do usuário, então ele vive em promptModelDownload().
 */
async function checkEnvironment() {
    // O navegador embutido do VS Code (Simple Browser) e o Electron em geral tem window.chrome
    // e ate um LanguageModel — mas é um stub que só devolve o texto de volta (echo).
    // Precisa ser barrado antes, senão a demo "funciona" e responde besteira.
    if (/Electron\//.test(navigator.userAgent)) {
        return {
            status: 'blocked',
            messages: [
                "⚠️ Você está no navegador embutido do VS Code (Electron), não no Chrome.",
                "Aqui o <code>LanguageModel</code> é um stub: ele não roda o Gemini Nano, apenas devolve o texto enviado (echo).",
                "Abra <b>http://127.0.0.1:8080</b> no Google Chrome ou Chrome Canary de verdade.",
            ],
        };
    }

    // @ts-ignore
    const isChrome = !!window.chrome;
    if (!isChrome) {
        return {
            status: 'blocked',
            messages: ["⚠️ Este recurso só funciona no Google Chrome ou Chrome Canary (versão recente)."],
        };
    }

    if (!('LanguageModel' in self)) {
        return {
            status: 'blocked',
            messages: [
                "⚠️ As APIs nativas de IA não estão ativas.",
                "Ative a seguinte flag em chrome://flags/:",
                "- Prompt API for Gemini Nano (chrome://flags/#prompt-api-for-gemini-nano)",
                "Depois reinicie o Chrome e tente novamente.",
            ],
        };
    }

    const availability = await LanguageModel.availability({ languages: ["pt"] });
    console.log('Language Model Availability:', availability);

    if (availability === 'available') {
        return { status: 'ready', availability, messages: [] };
    }

    if (availability === 'unavailable') {
        return {
            status: 'blocked',
            availability,
            messages: ["⚠️ O seu dispositivo não suporta modelos de linguagem nativos de IA."],
        };
    }

    // 'downloadable' | 'downloading'
    return { status: 'needs-download', availability, messages: [] };
}

function formatDownloadProgress(event) {
    // O Chrome mudou o formato do evento: hoje `loaded` é uma fração de 0 a 1
    // e `total` pode nem existir. Suporta os dois formatos.
    const ratio = event.total > 1 ? event.loaded / event.total : event.loaded;
    return `${Math.round(ratio * 100)}%`;
}

/**
 * Mostra um botão e só dispara o download dentro do clique.
 *
 * O Chrome exige "transient user activation" para iniciar/acompanhar o download do
 * Gemini Nano. Chamar LanguageModel.create() no carregamento da página resulta em:
 *   NotAllowedError: Requires a user gesture when availability is "downloading" or "downloadable".
 *
 * Resolve apenas quando o modelo estiver pronto; em caso de erro o botão volta a ficar
 * clicável para o usuário tentar de novo.
 */
function promptModelDownload(availability) {
    return new Promise((resolve) => {
        const output = elements.output;
        output.textContent = '';

        const info = document.createElement('div');
        info.innerHTML = availability === 'downloading'
            ? '⚠️ O Chrome já está baixando o modelo de IA (Gemini Nano).<br/>Clique abaixo para acompanhar o progresso e liberar a demo.'
            : '⚠️ O modelo de IA (Gemini Nano) ainda não está neste computador.<br/>O Chrome só inicia o download a partir de um clique seu — é um download grande, aguarde.';
        output.appendChild(info);

        const status = document.createElement('div');
        status.className = 'download-status';
        output.appendChild(status);

        const downloadButton = document.createElement('button');
        downloadButton.type = 'button';
        downloadButton.className = 'download-button';
        downloadButton.textContent = availability === 'downloading'
            ? 'Acompanhar download'
            : 'Baixar modelo agora';
        output.appendChild(downloadButton);

        downloadButton.addEventListener('click', () => {
            // IMPORTANTE: create() precisa ser a PRIMEIRA chamada do handler.
            // Qualquer `await` antes dela deixa a ativação transitória expirar
            // e o Chrome rejeita o download de novo.
            const creating = LanguageModel.create({
                expectedInputLanguages: ["pt"],
                monitor(m) {
                    m.addEventListener('downloadprogress', (event) => {
                        const progress = formatDownloadProgress(event);
                        console.log(`Downloaded ${progress}`);
                        status.textContent = `Baixando modelo: ${progress}`;
                    });
                },
            });

            downloadButton.disabled = true;
            status.textContent = 'Iniciando download... (o progresso também aparece no console do Chrome)';

            creating
                .then((session) => {
                    session.destroy();
                    status.textContent = '✅ Modelo baixado com sucesso!';
                    downloadButton.remove();
                    resolve();
                })
                .catch((error) => {
                    console.error('Error downloading model:', error);
                    status.textContent = `⚠️ Erro ao baixar o modelo: ${error.message}`;
                    downloadButton.disabled = false;
                });
        });
    });
}

/*
 * Limites padrao da spec, usados quando o Chrome nao expoe params()/capabilities().
 * defaultTemperature: 1 | maxTemperature: 2 | defaultTopK: 3 | maxTopK: 8
 */
const FALLBACK_PARAMS = {
    defaultTemperature: 1,
    maxTemperature: 2,
    defaultTopK: 3,
    maxTopK: 8,
};

/**
 * A superficie da Prompt API mudou entre versoes do Chrome:
 * capabilities() virou params(), e alguns builds nao tem nenhum dos dois.
 * Sem esta negociacao, `LanguageModel.params is not a function` derruba o app inteiro.
 */
async function getModelParams() {
    if (typeof LanguageModel.params === 'function') {
        const params = await LanguageModel.params();
        return { ...FALLBACK_PARAMS, ...params };
    }

    if (typeof LanguageModel.capabilities === 'function') {
        const capabilities = await LanguageModel.capabilities();
        console.warn('Este Chrome usa a API antiga: LanguageModel.capabilities()');
        return { ...FALLBACK_PARAMS, ...capabilities };
    }

    console.warn(
        'Este Chrome nao expoe params() nem capabilities(); usando os limites padrao da spec.',
        'Membros disponiveis em LanguageModel:',
        Object.getOwnPropertyNames(LanguageModel)
    );
    return { ...FALLBACK_PARAMS };
}

async function startApp() {
    const params = await getModelParams();
    console.log('Language Model Params:', params);

    elements.topK.max = params.maxTopK;
    elements.topK.min = 1;
    elements.topK.value = params.defaultTopK;
    elements.topKValue.textContent = params.defaultTopK;

    elements.temperatureValue.textContent = params.defaultTemperature;
    elements.temperature.max = params.maxTemperature;
    elements.temperature.min = 0;
    elements.temperature.value = params.defaultTemperature;

    elements.output.textContent = '';
    elements.button.disabled = false;

    setupEventListeners();
}

(async function main() {
    elements.year.textContent = new Date().getFullYear();
    elements.button.disabled = true;

    const environment = await checkEnvironment();

    if (environment.status === 'blocked') {
        elements.output.innerHTML = environment.messages.join('<br/>');
        return;
    }

    if (environment.status === 'needs-download') {
        await promptModelDownload(environment.availability);
    }

    try {
        await startApp();
    } catch (error) {
        // Sem isto o botao "Enviar" ficaria desabilitado em silencio, sem pista do motivo.
        console.error('Error starting the app:', error);
        elements.output.textContent = `⚠️ Nao foi possivel inicializar o modelo: ${error.message}`;
    }
})();
