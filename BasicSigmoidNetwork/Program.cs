using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BasicSigmoidNetwork
{
    class Program
    {
        static void Main(string[] args)
        {
            var function = new ActivationNetwork(new Sigmoid(), 0.05, 5, 5);
        }
    }
}
