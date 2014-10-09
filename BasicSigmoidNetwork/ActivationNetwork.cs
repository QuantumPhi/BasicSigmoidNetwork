using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BasicSigmoidNetwork
{
    class ActivationNetwork
    {
        public readonly int InCount { get; private set; }
        public readonly int HideCount { get; private set; }

        public double[] WeightsInput { get; private set; }
        public double[] WeightsHidden { get; private set; }

        public double Run(double[] input, double output)
        {
            Trace.Assert(input.Length == WeightsInput.Length);
            double sum = Enumerable
                .Range(0, input.Length)
                .Select(x =>
                {

                })
                .Sum();
            double sumHidden;
            return sumHidden > 0 ? 1 : 0;
        }
    }
}
