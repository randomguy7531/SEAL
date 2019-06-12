using Microsoft.Research.SEAL;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;

namespace UWPMPProjectTests
{
    [TestClass]
    public class TestNNBinary
    {
        [TestMethod]
        public void TestNNBinary_NoEncryption()
        {
            // this test exists simply to demonstrate that we have the capability of evaluating the ML model on some test data without the homomorphic encryption scheme
            // essentially, it serves to make sure the computations we will be performing on the encrypted data will be correct

            double[][] testX = new double[5][];
            testX[0] = new double[] { 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0 };
            testX[1] = new double[] { 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0 };
            testX[2] = new double[] { 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0 };
            testX[3] = new double[] { 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0 };
            testX[4] = new double[] { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0 };

            bool[] testYs = { true, false, true, false, false };

            var weights = NNConstants.GetNNBinaryWeights();
            var biases = NNConstants.GetNNBinaryBiases();
            List<double[][]> TestFFOutputs = new List<double[][]>();
            List<double[][]> TestActivationInputs = new List<double[][]>();
            for(int testI = 0; testI < testX.Length; testI ++)
            {
                var FFOutputs = NNConstants.GetEmptyFFVectors(weights);
                var ActivationInputs = NNConstants.GetEmptyFFVectors(weights);
                for (int layer = 0; layer < weights.Length; layer++)
                {
                    var numLayerInputs = weights[layer].Length;
                    var numLayerOutputs = weights[layer][0].Length;
                    for (int ffOutputIndex = 0; ffOutputIndex < numLayerOutputs; ffOutputIndex++)
                    {
                        double nodeOutput = 0.0;
                        for (int inputIndex = 0; inputIndex < numLayerInputs; inputIndex++)
                        {
                            if (layer == 0)
                            {
                                nodeOutput += testX[testI][inputIndex] * weights[layer][inputIndex][ffOutputIndex];
                            }
                            else
                            {
                                nodeOutput += FFOutputs[layer - 1][inputIndex] * weights[layer][inputIndex][ffOutputIndex];
                            }
                        }
                        nodeOutput += biases[layer][ffOutputIndex];
                        if (layer < (weights.Length - 1))
                        {
                            ActivationInputs[layer][ffOutputIndex] = nodeOutput;
                            nodeOutput *= nodeOutput;
                        }
                        else
                        {
                            ActivationInputs[layer][ffOutputIndex] = nodeOutput;
                            nodeOutput = NNConstants.Sigmoid(nodeOutput);
                        }
                        FFOutputs[layer][ffOutputIndex] = nodeOutput;
                    }
                }
                TestFFOutputs.Add(FFOutputs);
                TestActivationInputs.Add(ActivationInputs);
                Assert.AreEqual(FFOutputs[weights.Length - 1][0] >= 0.5, testYs[testI]);
            }
        }

        [TestMethod]
        public void TestNNBinary_Encryption()
        {
            double[][] testX = new double[5][];
            testX[0] = new double[] { 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0 };
            testX[1] = new double[] { 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0 };
            testX[2] = new double[] { 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0 };
            testX[3] = new double[] { 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0 };
            testX[4] = new double[] { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0 };

            bool[] testYs = { true, false, true, false, false };

            // setup the encryptor and other various components
            EncryptionParameters parms = new EncryptionParameters(SchemeType.CKKS);
            parms.PolyModulusDegree = 32768;
            parms.CoeffModulus = DefaultParams.CoeffModulus256(polyModulusDegree: 32768);
            SEALContext context = SEALContext.Create(parms);
            CKKSEncoder encoder = new CKKSEncoder(context);
            KeyGenerator keygen = new KeyGenerator(context);
            PublicKey publicKey = keygen.PublicKey;
            SecretKey secretKey = keygen.SecretKey;
            RelinKeys relinKeys = keygen.RelinKeys(decompositionBitCount: DefaultParams.DBCmax);
            Encryptor encryptor = new Encryptor(context, publicKey);
            Evaluator evaluator = new Evaluator(context);
            Decryptor decryptor = new Decryptor(context, secretKey);
            double scale = Math.Pow(2.0, 30);

            // encrypt the input features
            List<List<Ciphertext>> featureCiphers = new List<List<Ciphertext>>();
            for (int i = 0; i < testX.Length; i++)
            {
                List<Ciphertext> curFeatureCiphers = new List<Ciphertext>();
                foreach (var featureVal in testX[i])
                {
                    List<double> featureVector = new double[] { featureVal }.ToList();
                    Plaintext plain = new Plaintext();
                    encoder.Encode(featureVal, scale, plain);
                    Ciphertext encrypted = new Ciphertext();
                    encryptor.Encrypt(plain, encrypted);
                    curFeatureCiphers.Add(encrypted);
                }
                featureCiphers.Add(curFeatureCiphers);
            }

            /*
             * 
             * This represents the initial border between the client and the server
             * The only data that should cross this line is the public key computed above, and the cipher texts of the ML model features
             * 
             * */


            var weights = NNConstants.GetNNBinaryWeights();
            var weightsEncoded = NNConstants.GetWeightsPlaintext(encoder, scale, weights);
            var biases = NNConstants.GetNNBinaryBiases();
            List<Ciphertext[][]> TestFFOutputs = new List<Ciphertext[][]>();
            List<Ciphertext[][]> TestActivationInputs = new List<Ciphertext[][]>();
            for (int testI = 0; testI < testX.Length; testI++)
            {
                var FFOutputs = NNConstants.GetEmptyFFCipherVectors(weights);
                var ActivationInputs = NNConstants.GetEmptyFFCipherVectors(weights);
                for (int layer = 0; layer < weights.Length; layer++)
                {
                    var numLayerInputs = weights[layer].Length;
                    var numLayerOutputs = weights[layer][0].Length;
                    for (int ffOutputIndex = 0; ffOutputIndex < numLayerOutputs; ffOutputIndex++)
                    {
                        List<Ciphertext> sumInputs = new List<Ciphertext>();
                        //double nodeOutput = 0.0;
                        for (int inputIndex = 0; inputIndex < numLayerInputs; inputIndex++)
                        {
                            if (layer == 0)
                            {
                                Ciphertext multResult = new Ciphertext();
                                evaluator.MultiplyPlain(featureCiphers[testI][inputIndex], weightsEncoded[layer][inputIndex][ffOutputIndex], multResult);
                                sumInputs.Add(multResult);
                            }
                            else
                            {
                                Ciphertext multResult = new Ciphertext();
                                bool success = false;
                                int numTries = 0;
                                while (!success && numTries < 4)
                                {
                                    try
                                    {
                                        evaluator.MultiplyPlain(FFOutputs[layer - 1][inputIndex], weightsEncoded[layer][inputIndex][ffOutputIndex], multResult);
                                        success = true;
                                    }
                                    catch (Exception ex)
                                    {
                                        evaluator.RelinearizeInplace(FFOutputs[layer - 1][inputIndex], relinKeys);
                                        evaluator.RescaleToNextInplace(FFOutputs[layer - 1][inputIndex]);
                                        numTries++;
                                        if (numTries >= 4)
                                        {
                                            throw ex;
                                        }
                                    }
                                }
                                sumInputs.Add(multResult);
                            }
                        }
                        Ciphertext activationInput = new Ciphertext();
                        evaluator.AddMany(sumInputs, activationInput);
                        Plaintext biasPT = new Plaintext();
                        encoder.Encode(biases[layer][ffOutputIndex], activationInput.Scale, biasPT);
                        evaluator.AddPlainInplace(activationInput, biasPT);
                        Ciphertext activationOutput = new Ciphertext();
                        if (layer < (weights.Length - 1))
                        {
                            ActivationInputs[layer][ffOutputIndex] = activationInput;
                            bool success = false;
                            int numTries = 0;
                            while (!success && numTries < 4)
                            {
                                try
                                {
                                    evaluator.Square(activationInput, activationOutput);
                                    success = true;
                                }
                                catch (Exception ex)
                                {
                                    evaluator.RelinearizeInplace(activationInput, relinKeys);
                                    evaluator.RescaleToNextInplace(activationInput);
                                    numTries++;
                                    if (numTries >= 4)
                                    {
                                        throw ex;
                                    }
                                }
                            }
                        }
                        else
                        {
                            ActivationInputs[layer][ffOutputIndex] = activationInput;
                            // note that normally, we would do a sigmoid on this - however, we cant do a sigmoid
                            // in the encrypted space (we are limited to simple add/multiply
                            // so we just copy it to the output - the client will use a >/< check to determine class
                            activationOutput = activationInput;
                        }
                        FFOutputs[layer][ffOutputIndex] = activationOutput;
                    }
                }
                TestFFOutputs.Add(FFOutputs);
                TestActivationInputs.Add(ActivationInputs);
            }

            /*
             * 
             * This represents the next border between the client and the server
             * The only data that should cross this line is the encrypted output of running the ML model
             * In this case, the output decrypts to the input of a sigmoid - so the client will determine class membership
             * based on a > or < 0 boolean
             * 
             * */

            List<List<double>> predictions = new List<List<double>>();
            for (int testI = 0; testI < TestFFOutputs.Count; testI++)
            {
                Plaintext plainResult = new Plaintext();
                var encryptedResult = TestFFOutputs[testI][2][0];
                decryptor.Decrypt(encryptedResult, plainResult);
                List<double> result = new List<double>();
                encoder.Decode(plainResult, result);
                predictions.Add(result);
            }

            for (int testI = 0; testI < TestFFOutputs.Count; testI++)
            {
                var avgResult = predictions[testI].Average();
                bool prediction = false;
                if (avgResult >= 0)
                {
                    prediction = true;
                }
                Assert.AreEqual(prediction, testYs[testI]);
            }
        }
    }
}
