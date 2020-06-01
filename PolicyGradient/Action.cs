using AI;
using AI.ML.NeuralNetwork.CoreNNW;
using System;

namespace PolicyGradient
{
    public class Action
    {
        public Vector probabilities;
        Random random;

        public Action(Vector probabilities, Random random)
        {
            this.probabilities = probabilities;
            this.random = random;
        }
        public NNValue ToNNValue()
        {
            return new NNValue(probabilities);
        }

        public int GetAction()

        {
            while (true)
            {
                int index = random.Next(0, probabilities.Count);
                if (random.NextDouble() > 1.0 - probabilities[index]) {
                    return index;
                }
            }
        }


    }
}
