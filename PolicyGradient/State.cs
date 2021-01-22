using AI;
using AI.ML.NeuralNetwork.CoreNNW;

namespace PolicyGradient
{
    /// <summary>
    /// Состояние среды
    /// </summary>
    public class State
    {
        public NNValue Input { get; private set; }

        public State(Tensor input)
        {
            Input = new NNValue(input);
        }

        public State(Matrix input)
        {
            Input = new NNValue(input);
        }

        public State(Vector input)
        {
            Input = new NNValue(input);
        }

        public State(params double[] array)
        {
            Input = new NNValue(array);
        }
    }
}
