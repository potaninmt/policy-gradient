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

        private int SetAction(bool IsRnd)
        {
            if (!IsRnd)
            {
                index = probabilities.IndexMax();

                return index;
            }
            else
            {
                while (true)
                {
                    index = random.Next(0, probabilities.Count);
                    if (random.NextDouble() > 1.0 - probabilities[index])
                    {
                        return index;
                    }
                }
            }
        }

        public int GetAction()
        {
            return index;
        }


    }
}
