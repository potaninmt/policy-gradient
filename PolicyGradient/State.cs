using AI;
using AI.ML.NeuralNetwork.CoreNNW;

namespace PolicyGradient
{
    /// <summary>
    /// Состояние агента
    /// </summary>
    public class State
    {
        public NNValue input;

        public State(Tensor input)
        {
            this.input = new NNValue(input);
        }

        public State(Matrix input)
        {
            this.input = new NNValue(input);
        }

        public State(Vector input)
        {
            this.input = new NNValue(input);
        }

        public NNValue ToNNValue()
        {
            return input;
        }
    }
}
