using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace BasicSigmoidNetwork
{
    interface IActivationFunction
    {
        public double Function(double x);

        public double Derivative(double x);
    }

    public class Sigmoid : IActivationFunction
    {
        public double Function(double x)
        {
            return (1D) / (1 + Math.Exp(x));
        }

        public double Derivative(double x)
        {
            var e = Math.Exp(-x);
            return -(e) / (1 + 2D * e + Math.Pow(e, 2D));
        }
    }
}
