import tf from '@tensorflow/tfjs-node';


async function trainModel(xs, ys) {
    const model = tf.sequential();

    // primeira camada da rede:
    // entrada de 7 posições (idade, 3 cores + 3 localizações)

    // 80 neuronios = coloquei tudo isso porque tem pouca base de treino
    // quanto mais neuronio, mais complexidade a rede pode aprender
    // e consequentemente, mais processsamento ela vai usar, e mais tempo vai levar para treinar

    // a ReLU age como um filtro:
    // É como se ela deixasse somente os dados interessantes seguirem viagem na rede
    // se a informação chegou nesse neuronio é positiva, passa pra frente
    // se for zero ou negativa, pode jogar fora, nao vai servir pra nada, entao nao passa pra frente
    model.add(tf.layers.dense({ inputShape: [7], units: 80, activation: 'relu'}));
    // camada de saída: 3 neuronios, um pra cada categoria (premium, medium, basic)
    // activation: softmax normaliza a saida em uma probabilidade
    model.add(tf.layers.dense({ units: 3, activation: 'softmax' }));

    // Compilando o modelo
    // optimizer Adam ( Adaptive Moment Estimation)
    // é um treinador pessoal moderno para redes neurais
    // ele ajusta os pesos de forma eficiente e inteligente
    // aprender com historico de erros e acertos

    // loss: categoricalCrossentropy
    // ele compara o que o modelo "acha" ( os scores de cada categoria) com a resposta certa
    // a categoria premium sera sempre [1, 0, 0]

    // quanto mais distante da previsão do modelo da resposta correta
    // maior o erro (loss)
    // exemplo classico: classificação de imagens, recomendação e categorização de usuarios
    // qualquer coisa em que a resposta certa é "apenas uma entre várias possiveis"
    model.compile({ optimizer: 'adam', loss: 'categoricalCrossentropy', metrics: ['accuracy'] });

    // treinamento do modelo
    // verbose: desabilita o log interno e só usa o callbachk para mostrar o progresso
    // epochs: número de vezes que o modelo vai passar por todo o dataset de treino
    // shuffle: embaralha os dados a cada epoch para evitar que o modelo aprenda padrões específicos da ordem dos dados (viés)
    // callbacks: função que é chamada no final de cada epoch, para mostrar o progresso do treinamento
    await model.fit(xs, ys, { verbose: 0, epochs: 100, shuffle: true, callbacks: {
        onEpochEnd: (epoch, logs) => {
            // console.log(`Epoch ${epoch + 1}: loss = ${logs.loss.toFixed(4)}, accuracy = ${logs.acc.toFixed(4)}`);
        }
    } });

    return model;
}

async function predict(model, pessoa) {
    // transformar o array js para o tensor do tensorflow
    const tfInput = tf.tensor2d(pessoa);
    // faz a predição (output sera um vetor de 3 probabilidades)
    const prediction = model.predict(tfInput);
    // converte o tensor de volta para array js para ler o resultado
    const predictionArray = await prediction.array()
    return predictionArray[0].map((probabilidade, index) => ({ probabilidade, index }));
}
// Exemplo de pessoas para treino (cada pessoa com idade, cor e localização)
// const pessoas = [
//     { nome: "Erick", idade: 30, cor: "azul", localizacao: "São Paulo" },
//     { nome: "Ana", idade: 25, cor: "vermelho", localizacao: "Rio" },
//     { nome: "Carlos", idade: 40, cor: "verde", localizacao: "Curitiba" }
// ];

// Vetores de entrada com valores já normalizados e one-hot encoded
// Ordem: [idade_normalizada, azul, vermelho, verde, São Paulo, Rio, Curitiba]
// const tensorPessoas = [
//     [0.33, 1, 0, 0, 1, 0, 0], // Erick
//     [0, 0, 1, 0, 0, 1, 0],    // Ana
//     [1, 0, 0, 1, 0, 0, 1]     // Carlos
// ]

// Usamos apenas os dados numéricos, como a rede neural só entende números.
// tensorPessoasNormalizado corresponde ao dataset de entrada do modelo.
const tensorPessoasNormalizado = [
    [0.33, 1, 0, 0, 1, 0, 0], // Erick
    [0, 0, 1, 0, 0, 1, 0],    // Ana
    [1, 0, 0, 1, 0, 0, 1],     // Carlos
]

// Labels das categorias a serem previstas (one-hot encoded)
// [premium, medium, basic]
const labelsNomes = ["premium", "medium", "basic"]; // Ordem dos labels
const tensorLabels = [
    [1, 0, 0], // premium - Erick
    [0, 1, 0], // medium - Ana
    [0, 0, 1], // basic - Carlos
];

// Criamos tensores de entrada (xs) e saída (ys) para treinar o modelo
const inputXs = tf.tensor2d(tensorPessoasNormalizado)
const outputYs = tf.tensor2d(tensorLabels)


// quanto mais dado melhor
// assim o algoritmo consegue aprender melhor os padrões e fazer previsões mais precisas
const model = await trainModel(inputXs, outputYs);


const pessoa = { nome: "Luizinho", idade: 28, cor: "verde", localizacao: "Curitiba" }

// Normalizando a udade da pessoa usando o mesmo padrao do treino
// exemplo: idade_min = 25, idade_max = 40, então (28 - 25) / (40 - 25) = 0.2
const pessoaTensorNormalizado = [
    [
        0.2, // idade normalizada
        1, // azul
        0, // vermelho
        0, // verde
        1, // São Paulo
        0, // Rio
        0  // Curitiba
    ]
]

const predictions = await predict(model, pessoaTensorNormalizado);
const results = predictions
.sort((a, b) => b.probabilidade - a.probabilidade)
.map((p => `${labelsNomes[p.index]} ${(p.probabilidade * 100).toFixed(2)}%`)).join('\n');

console.log(`Previsões para ${pessoa.nome}:`);
console.log(results);
