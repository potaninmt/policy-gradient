using System;
using System.Collections.Generic;
using System.Linq;

using AI;
using AI.ML.Datasets;
using AI.ML.NeuralNetwork.CoreNNW;
using AI.ML.NeuralNetwork.CoreNNW.Layers;
using AI.ML.NeuralNetwork.CoreNNW.Loss;
using AI.ML.NeuralNetwork.CoreNNW.Models;
using AI.ML.NeuralNetwork.CoreNNW.Optimizers;
using AI.ML.NeuralNetwork.CoreNNW.Train;

namespace PolicyGradient
{
    public class MLAgent
    {
        public List<Generation> Generations { get; private set; }
        Trainer trainer;
        IOptimizer optimizer;

        double averageScore { get; set; }
        int degreesOfFreedom { get; set; }

        NNW model;
        IGraph graphForward, graphBackward;

        Random random;

        /// <summary>
        /// Конструктор
        /// </summary>
        /// <param name="model">Модель нейронной сети</param>
        /// <param name="degreesOfFreedom">Количество возможных действий</param>
        /// <param name="random">Генератор рандома</param>
        public MLAgent(NNW model, int degreesOfFreedom, Random random, IOptimizer optimizer = null)
        {
            Generations = new List<Generation>();
            graphForward = new GraphCPU(false);
            graphBackward = new GraphCPU(true);
            this.degreesOfFreedom = degreesOfFreedom;
            this.model = model;
            this.random = random;

            if (optimizer == null)
                optimizer = new Adam();

            this.optimizer = optimizer;

            trainer = new Trainer(graphBackward, TrainType.Online, optimizer);
        }

        /// <summary>
        /// Создать новое поколение
        /// </summary>
        /// <param name="life"></param>
        public void CreateGeneration()
        {
            Generations.Add(new Generation());
        }

        /// <summary>
        /// Добавить пару state, action для поколения
        /// </summary>
        /// <param name="state"></param>
        /// <param name="action"></param>
        public void AddCondition(State state, Action action)
        {
            var last = Generations.Last();
            last.Add(state, action);
        }

        /// <summary>
        /// Вернуть текущее поколение
        /// </summary>
        /// <returns></returns>
        public Generation GetGeneration()
        {
            return Generations.Last();
        }

        /// <summary>
        /// Получить i-ое поколение
        /// </summary>
        /// <param name="index">индекс поколения</param>
        /// <returns></returns>
        public Generation GetGeneration(int index)
        {
            return Generations[index];
        }

        /// <summary>
        /// Сгенерировать действие на воздействие(состояние среды)
        /// </summary>
        /// <param name="state"></param>
        /// <param name="isRnd">вероятностный подход</param>
        /// <returns></returns>
        public Action GetAction(State state, bool isRnd = true)
        {
            if(!(model.Layers.Last() is FeedForwardLayer))
            {
                throw new NotImplementedException();
            }
            else
            {
                var input = state.Input;
                var output = model.Activate(input, graphForward);
                var vector = new Vector(output.DataInTensor);

                return new Action(vector, random, isRnd);
            }

        }

        /// <summary>
        /// Обновить очки в текущей жизни
        /// </summary>
        /// <param name="score"></param>
        public void SetScore(double score)
        {
            Generations.Last().SetScore(score);
        }

        /// <summary>
        /// Обновить очки на i-той жизни
        /// </summary>
        /// <param name="index"></param>
        /// <param name="score"></param>
        public void SetScore(int index, double score)
        {
            Generations[index].SetScore(score);
        }

        /// <summary>
        /// Получить score последнего поколения
        /// </summary>
        /// <returns></returns>
        public double GetScore()
        {
            return Generations.Last().GetScore();
        }


        /// <summary>
        /// Получить score i-того поколения
        /// </summary>
        /// <param name="index">индекс жизни</param>
        /// <returns></returns>
        public double GetScore(int index)
        {
            return Generations[index].GetScore();
        }

        /// <summary>
        /// Получить знаковые оценки по поколениям
        /// </summary>
        /// <returns></returns>
        public Vector GetRewards()
        {
            return GetRewards(0, Generations.Count);
        }

        /// <summary>
        /// Получить знаковые оценки поколений [start; end)
        /// </summary>
        /// <param name="start"></param>
        /// <param name="end"></param>
        /// <returns></returns>
        public Vector GetRewards(int start, int end)
        {
            List<double> scores = new List<double>();
            for (int i = start; i < end; i++)
            {
                scores.Add(Generations[i].GetScore());
            }

            var vec = new Vector(scores);

            averageScore = scores.Average();

            vec -= averageScore;

            vec = vec.TransformVector(x => Math.Sign(x));

            return vec;
        }

        /// <summary>
        /// Очистить список поколений
        /// </summary>
        public void Clear()
        {
            Generations = new List<Generation>();
        }

        /// <summary>
        /// Обучить нейронную сеть на накопленных исследованиях
        /// </summary>
        /// <param name="countGenerations">На скольких жизней, начиная от последней, обучить модель</param>
        /// <param name="epochs">Количество эпох обучения. По умолчанию 1</param>
        /// <param name="learningRate">Норма обучения. По умолчанию 1e-3</param>
        /// <param name="trainType">Тип обучения. По умолчанию online</param>
        /// <param name="minLoss">ошибка, при которой обучение останавливается</param>
        /// <param name="optimizer">Оптимизатор. По умолчанию Adam</param>
        /// <param name="loss">Метрика ошибки. По умолчанию MSE</param>
        public void Train(int countGenerations = -1, int epochs = 1, float learningRate = 1e-3f, TrainType trainType = TrainType.Online, float minLoss = 0.0f, ILoss loss = null)
        {
            if (loss == null) loss = new LossMeanSqrSqrt();
            if (countGenerations == -1) countGenerations = Generations.Count;

            int start = Generations.Count - countGenerations;
            int end = start + countGenerations;
            if (end > Generations.Count)
                end = Generations.Count;
            Vector rewards = GetRewards(start, end);
            var inputs = new List<NNValue>();
            var outputs = new List<NNValue>();

            for (int i = 0; i < rewards.Count; i++)
            {
                var conditions = Generations[start+i].GetConditions();
                foreach(var condition in conditions)
                {
                    var state = condition.Item1;
                    var action = condition.Item2;
                    double p = 0.01;
                    

                    if (rewards[i] > 0)
                    {
                        inputs.Add(state.Input);

                        Vector outp = new Vector(degreesOfFreedom);
                        outp[action.index] = 1.0 - p;
                        outputs.Add(new NNValue(outp));
                    }
                    else
                    {
                        inputs.Add(state.Input);

                        int u = 0;
                        while ((u = random.Next(0, degreesOfFreedom)) == action.index) ;
                        Vector output = new Vector(degreesOfFreedom);
                        output[u] = 1.0 - p;
                        outputs.Add(new NNValue(output));
                    }
                }
            }

            Shuffle(inputs, outputs);

            #region Train
            DataSetNoReccurent dataSetNoReccurent = new DataSetNoReccurent(inputs.ToArray(), outputs.ToArray(), loss);
            trainer.Train(epochs, learningRate, model, dataSetNoReccurent, minLoss);
            #endregion
        }

        private void Shuffle(List<NNValue> inputs, List<NNValue> outputs)
        {
            var count = inputs.Count;
            for (int i = count - 1; i > 0; i--)
            {
                var j = random.Next(0, i + 1);
                NNValue t1 = inputs[i];
                NNValue t2 = outputs[i];

                inputs[i] = inputs[j];
                outputs[i] = outputs[j];

                inputs[j] = t1;
                outputs[j] = t2;
            }
        }

        /// <summary>
        /// Сохранить модель нейронной сети
        /// </summary>
        /// <param name="path">путь к модели</param>
        public void SaveModel(string path)
        {
            model.Save(path);
        }


        /// <summary>
        /// Загрузить модель обученной нейронной сети
        /// </summary>
        /// <param name="path">путь до модели</param>
        public void LoadModel(string path)
        {
            model = NNW.Load(path);
        }
    }
}
