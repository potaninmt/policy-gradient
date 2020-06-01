using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PolicyGradient
{
    public class Life
    {
        List<Tuple<State, Action>> pairs;
        double score;

        public Life()
        {
            pairs = new List<Tuple<State, Action>>();
        }

        public void Add(State state, Action action)
        {
            pairs.Add(new Tuple<State, Action>(state, action));
        }

        public void UpdateScore(double score)
        {
            this.score = score;
        }

        public double GetScore()
        {
            return score;
        }

        public List<Tuple<State, Action>> GetConditions()
        {
            return pairs;
        }
    }
}
