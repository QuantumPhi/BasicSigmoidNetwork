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
        public readonly double Alpha { get; private set; }
        public readonly int Inputs { get; private set; }
        public readonly int Hidden { get; private set; }

        public double[] WeightsInput { get; private set; }
        public double[] WeightsHidden { get; private set; }

        public readonly IActivationFunction activate { get; private set; }

        public ActivationNetwork(IActivationFunction function, double alpha, int input, int hidden)
        {
            this.activate = function;
            Alpha = alpha;
            WeightsInput = new double[(Inputs = input)];
            WeightsHidden = new double[(Hidden = hidden)];
        }

        public double Compute(double[] input)
        {
            Trace.Assert(input.Length == Inputs);

            double sum = activate.Function(Enumerable
                .Range(0, Inputs)
                .Select(x => WeightsInput[x] * input[x])
                .Sum());

            double sumHidden = activate.Derivative(Enumerable
                .Range(0, Hidden)
                .Select(x => WeightsHidden[x] * sum)
                .Sum());

            return sumHidden > 0 ? 1 : 0;
        }

        public double Train(double[] input, double output)
        {
            double error = 0.5 * Enumerable
                .Range(0, Inputs)
                .Select(x => Math.Pow(output - Compute(input), 2))
                .Sum();

            WeightsHidden = Enumerable
                .Range(0, Hidden)
                .Select(x => Alpha * (output - Compute(input)) * activate.Derivative(error) * input[x])
                .ToArray();

            WeightsInput = Enumerable
                .Range(0, Inputs)
                .Select(x =>
                {
                    double errorHidden = Enumerable
                        .Range(0, Hidden)
                        .Select(y => (output - Compute(input)) * activate.Derivative(error) * WeightsHidden[y])
                        .Sum();

                    double errorInput = errorHidden * activate.Function(errorHidden) * WeightsInput[x];

                    return errorInput;
                })
                .ToArray();

            return (1 - error) / 100;
        }
    }
}
