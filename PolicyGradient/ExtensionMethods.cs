using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PolicyGradient
{
    public static class ExtensionMethods
    {
        static Random random = new Random();
        public static IEnumerable<T> Shuffle<T>(this IEnumerable<T> collection)
        {
            var copy = collection.ToArray();
            var count = copy.Length;


            for (int i = count - 1; i > 0; i--)
            {
                var j = random.Next(0, i + 1);
                T t = copy[i];
                copy[i] = copy[j];
                copy[j] = t;
            }

            return copy;
        }
    }
}
