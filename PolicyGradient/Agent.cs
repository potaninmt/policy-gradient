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

        public void AddLife(Life life)
        {
            lifes.Add(life);
        }

        public void AddConditionToCurrentLife(State state, Action action)
        {
            lifes.Last().Add(state, action);
        }

        public Life GetCurrentLife()
        {
            return lifes.Last();
        }

        public Life GetLife(int index)
        {
            return lifes[index];
        }

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

        public void UpdateScoreToCurrentLife(double score)
        {
            lifes.Last().UpdateScore(score);
        }

        public void UpdateScore(int index, double score)
        {
            lifes[index].UpdateScore(score);
        }

        public double GetScoreCurrentLife()
        {
            return lifes.Last().GetScore();
        }

        public double GetScore(int index)
        {
            return lifes[index].GetScore();
        }

        public List<Life> GetLifes()
        {
            return lifes;
        }

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

        public void Remove()
        {
            lifes = new List<Life>();
        }

        public void Train(int epochs = 1, double learningRate = 1e-3, TrainType trainType = TrainType.Online, double minLoss = 0.0, IOptimizer optimizer = null, ILoss loss = null)
        {
            if (loss == null) loss = new LossMeanSqrSqrt();
            if (optimizer == null) optimizer = new Adam();

            Vector rewards = GetRewards();
            var inputs = new List<NNValue>();
            var outputs = new List<NNValue>();

            for (int i = 0; i < rewards.N; i++)
            {
                var conditions = lifes[i].GetConditions();
                for (int j = 0; j < conditions.Count; j++)
                {
                    var condition = conditions[j];
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

            for (int i = 0; i < inputs.Count; i++)
            {
                var a = random.Next(0, inputs.Count);
                var b = random.Next(0, inputs.Count);
                var temp1 = inputs[a];
                var temp2 = outputs[a];

                inputs[a] = inputs[b];
                outputs[a] = outputs[b];

                inputs[b] = temp1;
                outputs[b] = temp2;
            }

            DataSetNoReccurent dataSetNoReccurent = new DataSetNoReccurent(inputs.ToArray(), outputs.ToArray(), loss);
            Trainer trainer = new Trainer(graphBackward, trainType, optimizer);
            trainer.Train(epochs, learningRate, model, dataSetNoReccurent, minLoss);
        }
    }
}
