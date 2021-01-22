using AI;
using AI.ML.NeuralNetwork.CoreNNW;
using System;

namespace PolicyGradient
{
    public class Action
    {
        public Vector probabilities;
        public int index;
        public bool IsRnd;
        Random random;

        public Action(Vector probabilities, Random random, bool IsRnd = true)
        {
            this.probabilities = probabilities;
            this.random = random;
            this.IsRnd = IsRnd;
            this.index = SetAction(IsRnd);
        }

        public NNValue ToNNValue()
        {
            return new NNValue(probabilities);
        }

        public int GetAction()
        {
            return index;
        }

        private int SetAction(bool IsRnd)
        {
            if (!IsRnd)
            {
                index = probabilities.IndexMax();

                return index;
            }
            else
            {
                var p = random.NextDouble();
                var values = probabilities / probabilities.Sum();

                double sum = 0.0;
                for (int i = 0; i < values.Count; i++)
                {
                    values[i] = sum + values[i];
                    sum += values[i];
                }

                for (int i = 0; i < values.Count; i++)
                {
                    var last = i == 0 ? -1e-8 : values[i - 1];
                    if (p > last && p <= values[i])
                    {
                        index = i;
                        return index;
                    }
                }

                throw new Exception("prob error");
            }
        }
    }
}
