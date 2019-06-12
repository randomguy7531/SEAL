using Microsoft.Research.SEAL;
using System;
using System.Collections.Generic;
using System.Text;

namespace UWPMPProjectTests
{
    public class NNConstants
    {
        public static double[][][] GetNNBinaryWeights()
        {
            double[][] layer1Weights = new double[10][];
            layer1Weights[0] = new double[] { 1.08198404e-01, -1.54332623e-01, -2.91649938e-01, 1.28086433e-01, 1.54068530e-01, 2.28516422e-02, -2.07995266e-01, -1.25956059e-01, -2.17135772e-01, -2.83703446e-01, -1.94512382e-01, 9.12313312e-02 };
            layer1Weights[1] = new double[] { 4.82883714e-02, -9.89344791e-02, -8.72572139e-02, 5.62667772e-02, 7.08040372e-02, 3.89956445e-01, -4.65919942e-01, 2.59364415e-02, 6.38050139e-02, -9.82821137e-02, 6.39209850e-03, 1.29682776e-02 };
            layer1Weights[2] = new double[] { 3.74554889e-03, -2.47994334e-01, -2.79475957e-01, 2.50520408e-01, 2.29981974e-01, -1.47635490e-01, 3.62072974e-01, -2.42490619e-02, -1.81615144e-01, -2.80776799e-01, -2.59705603e-01, 1.48322970e-01 };
            layer1Weights[3] = new double[] { -2.34480128e-01, -5.28591037e-01, -6.04451954e-01, 4.30923849e-01, 5.19501626e-01, 3.28817636e-01, -1.47031903e-01, 3.56593698e-01, -3.11986744e-01, -5.89885831e-01, -4.32990134e-01, 4.65461850e-01 };
            layer1Weights[4] = new double[] { 1.14462584e-01, -1.07746743e-01, -8.40409398e-02, 5.84131703e-02, 7.88528845e-02, -6.36488348e-02, 2.15478554e-01, -2.43588239e-02, 5.60704386e-03, -7.30310529e-02, -3.97535600e-02, 4.15966474e-02 };
            layer1Weights[5] = new double[] { 1.94707543e-01, -7.25551024e-02, -1.11244461e-02, 7.20724091e-02, 4.43527810e-02, 2.69690007e-01, -2.87631452e-01, -9.51419398e-03, 5.71903214e-03, -5.83580770e-02, -3.31406705e-02, -1.56303421e-02 };
            layer1Weights[6] = new double[] { -1.00759625e-01, -2.22447261e-01, -1.56235054e-01, 2.27349877e-01, 2.82248646e-01, 5.89620904e-04, -4.46591265e-02, 8.74447078e-02, -1.64863139e-01, -2.14786142e-01, -1.71541035e-01, 2.14616612e-01 };
            layer1Weights[7] = new double[] { -1.09153047e-01, -2.82173157e-01, -1.35632053e-01, 2.16750249e-01, 2.57989109e-01, -2.36393660e-01, 1.88968241e-01, 1.15583111e-02, 1.97238848e-02, -1.55143932e-01, -5.72882034e-02, 2.04373688e-01 };
            layer1Weights[8] = new double[] { 1.93230510e-01, -5.85671440e-02, -6.32283166e-02, 1.24272443e-01, 9.83968675e-02, -2.34835595e-01, 1.66391075e-01, -1.01717383e-01, -2.54909962e-01, -8.85061771e-02, -1.30221009e-01, 6.85390979e-02 };
            layer1Weights[9] = new double[] { 1.86165929e-01, -2.05264792e-01, -9.49981436e-02, 1.62949681e-01, 2.51603067e-01, 2.86010534e-01, -1.82634413e-01, -1.16780117e-01, 1.12821748e-02, -1.39998659e-01, -5.77335507e-02, 2.43607551e-01 };

            double[][] layer2Weights = new double[12][];
            layer2Weights[0] = new double[] { 0.36188033, -0.33404708, 0.36808565, -0.35339352, 0.28110757, -0.06881063, 0.08834275, -0.3470543 };
            layer2Weights[1] = new double[] { -0.10862389, 0.1318869, -0.07577699, 0.13178398, -0.12880333, 0.19090506, -0.2975456, 0.14948705 };
            layer2Weights[2] = new double[] { -0.1808939, 0.1640419, -0.14266983, 0.17224377, -0.2114963, 0.31100583, -0.2457134, 0.2091506 };
            layer2Weights[3] = new double[] { -0.05877648, 0.05653473, -0.06049068, 0.05488992, -0.02428113, 0.23792207, -0.24422789, 0.07711071 };
            layer2Weights[4] = new double[] { -0.08575825, 0.07740829, -0.07523912, 0.09320975, -0.07198334, 0.2586549, -0.3118792, 0.04044494 };
            layer2Weights[5] = new double[] { 0.18763687, -0.16652411, 0.16666237, -0.23470218, 0.24914263, 0.16839395, -0.18909429, -0.2646694 };
            layer2Weights[6] = new double[] { 0.17107785, -0.2227157, 0.12122927, -0.21747419, 0.1848058, 0.18342729, -0.2547539, -0.21156688 };
            layer2Weights[7] = new double[] { 0.32296315, -0.33974877, 0.28254104, -0.36070928, 0.31019846, -0.15844429, 0.11699786, -0.27177042 };
            layer2Weights[8] = new double[] { 0.05060432, -0.06319131, 0.00201767, 0.00487328, 0.06533872, 0.18092534, -0.23294064, -0.04633536 };
            layer2Weights[9] = new double[] { -0.08643375, 0.13019231, -0.08043572, 0.11887724, -0.13155636, 0.26360306, -0.31203368, 0.13526444 };
            layer2Weights[10] = new double[] { -0.04849133, -0.00333728, -0.05743499, -0.01845557, -0.04396923, 0.16758396, -0.2608221, 0.01017937 };
            layer2Weights[11] = new double[] { -0.0154861, 0.07981848, -0.03590481, 0.05785968, -0.0922742, 0.19720666, -0.26465055, 0.06652021 };

            double[][] layer3Weights = new double[8][];
            layer3Weights[0] = new double[] { -0.5179071 };
            layer3Weights[1] = new double[] { -0.52756804 };
            layer3Weights[2] = new double[] { -0.47361284 };
            layer3Weights[3] = new double[] { -0.50819 };
            layer3Weights[4] = new double[] { -0.5068695 };
            layer3Weights[5] = new double[] { 0.24205086 };
            layer3Weights[6] = new double[] { 0.26278087 };
            layer3Weights[7] = new double[] { -0.5649978 };

            double[][][] toReturn = new double[3][][];
            toReturn[0] = layer1Weights;
            toReturn[1] = layer2Weights;
            toReturn[2] = layer3Weights;

            return toReturn;
        }

        public static double[][] GetNNBinaryBiases()
        {
            double[][] toReturn = new double[3][];
            double[] layer1Biases = { 0.5436792, -0.05232789, -0.03477787, 0.04422415, 0.08905727, -0.83102155, 0.81963444, -0.4861503, -0.14125478, -0.08763575, -0.0951765, 0.05849119 };
            double[] layer2Biases = { 0.7739916, -0.7730737, 0.7500151, -0.7737239, 0.7784327, -0.10003804, 0.09053684, -0.78069234 };
            double[] layer3Biases = { -0.5709162 };
            toReturn[0] = layer1Biases;
            toReturn[1] = layer2Biases;
            toReturn[2] = layer3Biases;
            return toReturn;
        }

        public static Plaintext[][][] GetWeightsPlaintext(CKKSEncoder encoder, double scale, double[][][] weights)
        {
            Plaintext[][][] toReturn = new Plaintext[weights.Length][][];
            for(int i = 0; i < weights.Length; i++)
            {
                Plaintext[][] layerWeightsPlaintext = new Plaintext[weights[i].Length][];
                for(int j = 0; j < weights[i].Length; j++)
                {
                    Plaintext[] layerNodeWeightsPlaintext = new Plaintext[weights[i][j].Length];
                    for(int k = 0; k < weights[i][j].Length; k++)
                    {
                        Plaintext weightPT = new Plaintext();
                        encoder.Encode(weights[i][j][k], scale, weightPT);
                        layerNodeWeightsPlaintext[k] = weightPT;
                    }
                    layerWeightsPlaintext[j] = layerNodeWeightsPlaintext;
                }
                toReturn[i] = layerWeightsPlaintext;
            }
            return toReturn;
        }

        public static double[][] GetEmptyFFVectors(double[][][] weights)
        {
            var numLayers = weights.Length;
            double[][] toReturn = new double[numLayers][];
            for(int i = 0; i < numLayers; i++)
            {
                var numLayerInputs = weights[i].Length;
                var numLayerOutputs = weights[i][0].Length;
                double[] layerOutputs = new double[numLayerOutputs];
                toReturn[i] = layerOutputs;
            }
            return toReturn;
        }

        public static Ciphertext[][] GetEmptyFFCipherVectors(double[][][] weights)
        {
            var numLayers = weights.Length;
            Ciphertext[][] toReturn = new Ciphertext[numLayers][];
            for (int i = 0; i < numLayers; i++)
            {
                var numLayerInputs = weights[i].Length;
                var numLayerOutputs = weights[i][0].Length;
                Ciphertext[] layerOutputs = new Ciphertext[numLayerOutputs];
                toReturn[i] = layerOutputs;
            }
            return toReturn;
        }

        public static double Sigmoid(double value)
        {
            var k = Math.Exp(value);
            return k / (1.0 + k);
        }
    }
}
