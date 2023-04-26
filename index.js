const tf = require("@tensorflow/tfjs");
const use = require("@tensorflow-models/universal-sentence-encoder");
const axios = require("axios");
const pdfParse = require("pdf-parse");
const openai = require("openai");
const fs = require("fs");
const util = require("util");
const streamPipeline = util.promisify(require("stream").pipeline);

class SemanticSearch {
  constructor() {
    this.use = null;
    this.fitted = false;
  }

  async loadModel() {
    this.use = await use.load();
  }

  async fit(data, batch = 1000, nNeighbors = 5) {
    this.data = data;
    this.embeddings = await this.getTextEmbedding(data, batch);
    nNeighbors = Math.min(nNeighbors, this.embeddings.length);
    this.nn = new NearestNeighbors(nNeighbors);
    this.nn.fit(this.embeddings);
    this.fitted = true;
  }

  async call(text, returnData = true) {
    const inpEmb = await this.use.embed([text]);
    const neighbors = this.nn.kneighbors(inpEmb.arraySync(), returnData);

    if (returnData) {
      return neighbors.map((i) => this.data[i]);
    } else {
      return neighbors;
    }
  }

  async getTextEmbedding(texts, batch = 1000) {
    const embeddings = [];
    for (let i = 0; i < texts.length; i += batch) {
      const textBatch = texts.slice(i, i + batch);
      const embBatch = await this.use.embed(textBatch);
      embeddings.push(embBatch);
    }
    const finalEmbeddings = tf.concat(embeddings, 0);
    return finalEmbeddings.arraySync();
  }
}

class NearestNeighbors {
  constructor(nNeighbors) {
    this.nNeighbors = nNeighbors;
  }

  fit(embeddings) {
    this.embeddings = embeddings;
  }

  kneighbors(embedding, returnData = false) {
    const distances = this.embeddings.map((e) => this.distance(e, embedding));
    const sortedIndices = this.argsort(distances);
    const neighbors = sortedIndices.slice(0, this.nNeighbors);

    if (returnData) {
      return neighbors.map((i) => this.embeddings[i]);
    } else {
      return neighbors;
    }
  }

  distance(a, b) {
    return Math.sqrt(a.reduce((sum, _, i) => sum + (a[i] - b[i]) ** 2, 0));
  }

  argsort(array) {
    return array.map((_, i) => i).sort((a, b) => array[a] - array[b]);
  }
}

const recommender = new SemanticSearch();

(async () => {
  await recommender.loadModel();
})();

const downloadPDF = async (url, outputPath) => {
  const response = await axios.get(url, { responseType: "arraybuffer" });
  require("fs").writeFileSync(outputPath, new Buffer.from(response.data), "binary");
};

const preprocess = (text) => {
  text = text.replace(/\n/g, " ");
  text = text.replace(/\s+/g, " ");
  return text;
};

const pdfToText = async (path, startPage = 1, endPage = null) => {
  const data = await pdfParse(require("fs").readFileSync(path));

  if (endPage === null) {
    endPage = data.numpages;
  }

  const textList = [];

  for (let i = startPage - 1; i < endPage; i++) {
    const text = data.text;
    const preprocessedText = preprocess(text);
    textList.push(preprocessedText);
  }

  return textList;
};

const textToChunks = (texts, wordLength = 150, startPage = 1) => {
  const textToks = texts.map((t) => t.split(" "));
  const page_nums = [];
  const chunks = [];

  for (let idx = 0; idx < textToks.length; idx++) {
    const words = textToks[idx];
    for (let i = 0; i < words.length; i += wordLength) {
      let chunk = words.slice(i, i + wordLength);
      if (i + wordLength > words.length && chunk.length < wordLength && textToks.length !== idx + 1) {
        textToks[idx + 1] = chunk.concat(textToks[idx + 1]);
        continue;
      }
      chunk = chunk.join(" ").trim();
      chunk = `[${idx + startPage}] "${chunk}"`;
      chunks.push(chunk);
    }
  }

  return chunks;
};

// You need to set up your OpenAI API key here
openai.apiKey = "your_openai_api_key";

const generateText = async (openAIKey, prompt, engine = "text-davinci-002") => {
  openai.apiKey = openAIKey;
  const completions = await openai.Completion.create({
    engine: engine,
    prompt: prompt,
    max_tokens: 512,
    n: 1,
    stop: null,
    temperature: 0.7,
  });

  const message = completions.choices[0].text;
  return message;
};

async function generateAnswer(question, openAIKey) {
  const topNChunks = await recommender.call(question);
  let prompt = "";
  prompt += "search results:\n\n";
  for (const c of topNChunks) {
    prompt += c + "\n\n";
  }

  prompt +=
    "Instructions: Compose a comprehensive reply to the query using the search results given. " +
    "Cite each reference using [ Page Number] notation (every result has this number at the beginning). " +
    "Citation should be done at the end of each sentence. If the search results mention multiple subjects " +
    "with the same name, create separate answers for each. Only include information found in the results and " +
    "don't add any additional information content. If the text does not relate to the query, simply state 'Text Not Found in PDF'. Ignore outlier " +
"search results which have nothing to do with the question. Only answer what is asked. The " +
"answer should be short and concise. Answer step-by-step. \n\nQuery: {question}\nAnswer: ";

prompt += Query: ${question}\nAnswer:;
const answer = await generateText(openAIKey, prompt, "text-davinci-002");
return answer;
};

async function questionAnswer(url, file, question, openAIKey) {
  if (openAIKey.trim() === "") {
    return "[ERROR]: Please enter your Open AI Key. Get your key here: https://platform.openai.com/account/api-keys";
  }
  if (url.trim() === "" && file === null) {
    return "[ERROR]: Both URL and PDF are empty. Provide at least one.";
  }

  if (url.trim() !== "" && file !== null) {
    return "[ERROR]: Both URL and PDF are provided. Please provide only one (either URL or PDF).";
  }

  if (url.trim() !== "") {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Unexpected response ${response.statusText}`);
    }
    await streamPipeline(response.body, fs.createWriteStream("corpus.pdf"));
    await loadRecommender("corpus.pdf");
  } else {
    // You should handle file upload and processing as done in the original Python code
    const oldFileName = file.name;
    const fileName = file.name.slice(0, -12) + file.name.slice(-4);
    fs.renameSync(oldFileName, fileName);
    await loadRecommender(fileName);
  }

  if (question.trim() === "") {
    return "[ERROR]: Question field is empty";
  }

  return await generateAnswer(question, openAIKey);
}
