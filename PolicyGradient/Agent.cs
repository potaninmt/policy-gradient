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
    public class Agent
    {
        List<Life> lifes;

        public double averageScore { get; set; }
        public int degreesOfFreedom { get; private set; }

        public NNW model;
        public IGraph graphForward, graphBackward;

        Random random;
        public Agent(NNW model, int degreesOfFreedom, Random random)
        {
            lifes = new List<Life>();
            graphForward = new GraphCPU(false);
            graphBackward = new GraphCPU(true);

            this.degreesOfFreedom = degreesOfFreedom;
            this.model = model;
            this.random = random;
        }

        /// <summary>
        /// Начать новую жизнь
        /// </summary>
        /// <param name="life"></param>
        public void AddLife(Life life)
        {
            lifes.Add(life);
        }

        /// <summary>
        /// Добавить state и action на текущей жизни
        /// </summary>
        /// <param name="state"></param>
        /// <param name="action"></param>
        public void AddConditionToCurrentLife(State state, Action action)
        {
            lifes.Last().Add(state, action);
        }

        /// <summary>
        /// Вернуть текущую жизнь
        /// </summary>
        /// <returns></returns>
        public Life GetCurrentLife()
        {
            return lifes.Last();
        }

        /// <summary>
        /// Получить i-тую жизнь
        /// </summary>
        /// <param name="index">индекс жизни</param>
        /// <returns></returns>
        public Life GetLife(int index)
        {
            return lifes[index];
        }

        /// <summary>
        /// Сгенерировать действие на воздействие(состояние среды)
        /// </summary>
        /// <param name="state"></param>
        /// <returns></returns>
        public Action GetAction(State state)
        {
            if(!(model.Layers.Last() is FeedForwardLayer))
            {
                throw new NotImplementedException();
            }
            else
            {
                var input = state.ToNNValue();
                var output = model.Activate(input, graphForward);
                var vector = new Vector(output.DataInTensor);

                return new Action(vector, random);
            }

        }

        /// <summary>
        /// Обновить очки в текущей жизни
        /// </summary>
        /// <param name="score"></param>
        public void UpdateScoreToCurrentLife(double score)
        {
            lifes.Last().UpdateScore(score);
        }

        /// <summary>
        /// Обновить очки на i-той жизни
        /// </summary>
        /// <param name="index"></param>
        /// <param name="score"></param>
        public void UpdateScore(int index, double score)
        {
            lifes[index].UpdateScore(score);
        }

        /// <summary>
        /// Получить очки за текущую жизнь
        /// </summary>
        /// <returns></returns>
        public double GetScoreCurrentLife()
        {
            return lifes.Last().GetScore();
        }


        /// <summary>
        /// Получить очки на i-той жизни
        /// </summary>
        /// <param name="index">индекс жизни</param>
        /// <returns></returns>
        public double GetScore(int index)
        {
            return lifes[index].GetScore();
        }

        /// <summary>
        /// Получить лист жизней
        /// </summary>
        /// <returns></returns>
        public List<Life> GetLifes()
        {
            return lifes;
        }

        /// <summary>
        /// Получить знаковые оценки по всем прожитым жизням
        /// </summary>
        /// <returns></returns>
        public Vector GetRewards()
        {
            List<double> scores = new List<double>();
            foreach (var life in lifes)
            {
                scores.Add(life.GetScore());
            }

            var vec = new Vector(scores);

            averageScore = scores.Average();

            vec -= averageScore;

            vec = vec.TransformVector(x => Math.Sign(x));

            return vec;
        }

        /// <summary>
        /// Получить знаковые оценки по прожитым жизням [start; end)
        /// </summary>
        /// <param name="start"></param>
        /// <param name="end"></param>
        /// <returns></returns>
        public Vector GetRewards(int start, int end)
        {
            List<double> scores = new List<double>();
            for (int i = start; i < end; i++)
            {
                scores.Add(lifes[i].GetScore());
            }

            var vec = new Vector(scores);

            averageScore = scores.Average();

            vec -= averageScore;

            vec = vec.TransformVector(x => Math.Sign(x));

            return vec;
        }

        /// <summary>
        /// Очистить список жизней
        /// </summary>
        public void Remove()
        {
            lifes = new List<Life>();
        }

        /// <summary>
        /// Обучить нейронную сеть на накопленных исследованиях
        /// </summary>
        /// <param name="countLifes">На скольких жизней, начиная от последней, обучить модель</param>
        /// <param name="epochs">Количество эпох обучения. По умолчанию 1</param>
        /// <param name="learningRate">Норма обучения. По умолчанию 1e-3</param>
        /// <param name="trainType">Тип обучения. По умолчанию online</param>
        /// <param name="minLoss">ошибка, при которой обучение останавливается</param>
        /// <param name="optimizer">Оптимизатор. По умолчанию Adam</param>
        /// <param name="loss">Метрика ошибки. По умолчанию MSE</param>
        public void Train(int countLifes = 50, int epochs = 1, double learningRate = 1e-3, TrainType trainType = TrainType.Online, double minLoss = 0.0, IOptimizer optimizer = null, ILoss loss = null)
        {
            if (loss == null) loss = new LossMeanSqrSqrt();
            if (optimizer == null) optimizer = new Adam();

            int start = lifes.Count - countLifes;
            Vector rewards = GetRewards(start, lifes.Count);
            var inputs = new List<NNValue>();
            var outputs = new List<NNValue>();

            for (int i = 0; i < rewards.N; i++)
            {
                var conditions = lifes[start+i].GetConditions();
                foreach(var condition in conditions)
                {
                    var state = condition.Item1;
                    var action = condition.Item2;

                    inputs.Add(state.ToNNValue());

                    if (rewards[i] > 0)
                    {
                        outputs.Add(new NNValue(action.probabilities.MaxOutVector().TransformVector(x => (x == -1) ? 0 : 1)));
                    }
                    else
                    {
                        outputs.Add(new NNValue((1.0 - action.probabilities).MaxOutVector().TransformVector(x => (x == -1) ? 0 : 1)));
                    }
                }
            }

            #region Shuffle
            for (int i = start; i < inputs.Count; i++)
            {
                var a = random.Next(start, inputs.Count);
                var b = random.Next(start, inputs.Count);
                var temp1 = inputs[a];
                var temp2 = outputs[a];

                inputs[a] = inputs[b];
                outputs[a] = outputs[b];

                inputs[b] = temp1;
                outputs[b] = temp2;
            }
            #endregion

            #region Train
            DataSetNoReccurent dataSetNoReccurent = new DataSetNoReccurent(inputs.ToArray(), outputs.ToArray(), loss);
            Trainer trainer = new Trainer(graphBackward, trainType, optimizer);
            trainer.Train(epochs, learningRate, model, dataSetNoReccurent, minLoss);
            #endregion
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
