using MathNet.Numerics.LinearAlgebra.Single;
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
        public readonly double alpha;
        public readonly int inputs;
        public readonly int hidden;
        //remove {get;private set;}
        public double[] WeightsInput { get; private set; }
        public double[] WeightsHidden { get; private set; }

        private readonly IActivationFunction activate;

        public ActivationNetwork(IActivationFunction function, double alpha, int input, int hidden)
        {
            this.activate = function;
            alpha = alpha;
            WeightsInput = new double[(inputs = input)];
            WeightsHidden = new double[(hidden = hidden)];
        }

        public double Compute(double[] input)
        {
            Trace.Assert(input.Length == inputs);

            //individual activation
            double sum = activate.Function(Enumerable
                .Range(0, inputs)
                .Select(x => WeightsInput[x] * input[x])
                .Sum());

            //individual activation
            // still normal function
            double sumHidden = activate.Derivative(Enumerable
                .Range(0, hidden)
                .Select(x => WeightsHidden[x] * sum)
                .Sum());

            return sumHidden > 0 ? 1 : 0;
        }

        public double Train(double[] input, double output)
        {
            double error = 0.5 * Enumerable
                .Range(0, inputs)
                .Select(x => Math.Pow(output - Compute(input), 2))
                .Sum();

            WeightsHidden = Enumerable
                .Range(0, hidden)
                .Select(x => alpha * activate.Derivative(error * input[x]))
                .ToArray();

            double errorHidden = Enumerable
                .Range(0, hidden)
                .Select(y => WeightsHidden[y] + activate.Derivative(WeightsHidden[y] * error * alpha))
                .Sum();

            WeightsInput = Enumerable
                .Range(0, inputs)
                .Select(x =>
                {
                    double errorInput = errorHidden * activate.Function(errorHidden) * WeightsInput[x];
                    return errorInput;
                })
                .ToArray();

            return error;
        }
    }
}
