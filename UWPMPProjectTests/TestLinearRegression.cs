using Microsoft.Research.SEAL;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;

namespace UWPMPProjectTests
{
    [TestClass]
    public class TestLinearRegression
    {
        [TestMethod]
        public void TestLinearRegression_NoEncryption()
        {
            // these are the feature sets we will be encrypting and getting
            // the ML model results on. The plaintext versions of these values
            // should (in the encryped scheme) not be known by the evaluator

            double[][] testX = new double[6][];
            testX[0] = new double[] { 43.45089541, 53.15091973, 53.07708932, 93.65456934, 65.23330105, 69.34856259, 62.91649012, 35.28814156, 108.1002775, 100.1735266 };
            testX[1] = new double[] { 51.59952075, 99.48561775, 95.75948428, 126.6533636, 142.5235433, 90.97955769, 43.66586306, 85.31957886, 62.57644682, 66.12458533 };
            testX[2] = new double[] { 94.77026243, 71.51229208, 85.33271407, 69.58347566, 107.8693045, 101.6701889, 89.88200921, 54.93440139, 105.5448532, 72.07947083 };
            testX[3] = new double[] { 89.53820766, 100.199631, 86.19911875, 85.88717675, 33.92249944, 80.47113937, 65.34411148, 89.70004394, 75.00778202, 122.3514331 };
            testX[4] = new double[] { 96.86101454, 97.54597612, 122.9960987, 86.1281547, 115.5539807, 107.888993, 65.51660154, 74.17007885, 48.04727402, 93.56952259 };
            testX[5] = new double[] { 91.75121904, 121.2115065, 62.92763365, 99.4343452, 70.420912, 88.0580948, 71.82993308, 80.49171244, 87.11321454, 100.1459868 };

            

            // This is the 'evaluator' section
            // this is not a part of the client and would, in a cloud based solution, be run on the server
            // the server should not know the values of the input features, but it will do math on them
            // likewise, the client would not know the various weights and parameters, but will receive a result
            double[] weights_model =
                    {0.0921533,
                    0.14545279,
                    0.11066622,
                    0.06119513,
                    0.10041948,
                    0.19091597,
                    0.07359407,
                    0.04503237,
                    0.10848583,
                    0.10092494};
            double intercept_model = -2.484390163425502;
            double[] weights_groundTruth = 
                { 0.1,0.15,0.1,0.05,0.1,0.2,0.05,0.05,0.1,0.1};

            List<double> predictions = new List<double>();
            for (int i = 0; i < testX.Length; i++)
            {
                double[] xFeatures = testX[i];

                var score = 0.0;
                for (int j = 0; j < xFeatures.Length; j++)
                {
                    score += weights_model[j] * xFeatures[j];
                }
                predictions.Add(score + intercept_model);
            }

            // this is the output section. In practice, we would not have access to these values (as they represent the ground truth)
            // We are using them here merely to demonstrate that we can properly recover the proper ML model output
            double[] yGroundTruth = { 68.43881952, 87.75905253, 88.34053641, 83.87264322, 95.20322583, 89.61704108 };
            double[] yTest = { 68.702934, 87.28860458, 88.40827187, 83.24001674, 94.53137951, 89.00229455 };
            const double EPSILON = 1e-4;
            for (int i = 0; i < predictions.Count; i++)
            {
                var absDelta = Math.Abs(predictions[i]- yTest[i]);
             Assert.IsTrue(absDelta < EPSILON);
            }
        }

        [TestMethod]
        public void TestLinearRegression_Encryption()
        {
            // these are the feature sets we will be encrypting and getting
            // the ML model results on. The plaintext versions of these values
            // should (in the encryped scheme) not be known by the evaluator

            double[][] testX = new double[6][];
            testX[0] = new double[] { 43.45089541, 53.15091973, 53.07708932, 93.65456934, 65.23330105, 69.34856259, 62.91649012, 35.28814156, 108.1002775, 100.1735266 };
            testX[1] = new double[] { 51.59952075, 99.48561775, 95.75948428, 126.6533636, 142.5235433, 90.97955769, 43.66586306, 85.31957886, 62.57644682, 66.12458533 };
            testX[2] = new double[] { 94.77026243, 71.51229208, 85.33271407, 69.58347566, 107.8693045, 101.6701889, 89.88200921, 54.93440139, 105.5448532, 72.07947083 };
            testX[3] = new double[] { 89.53820766, 100.199631, 86.19911875, 85.88717675, 33.92249944, 80.47113937, 65.34411148, 89.70004394, 75.00778202, 122.3514331 };
            testX[4] = new double[] { 96.86101454, 97.54597612, 122.9960987, 86.1281547, 115.5539807, 107.888993, 65.51660154, 74.17007885, 48.04727402, 93.56952259 };
            testX[5] = new double[] { 91.75121904, 121.2115065, 62.92763365, 99.4343452, 70.420912, 88.0580948, 71.82993308, 80.49171244, 87.11321454, 100.1459868 };

            // setup the encryptor and other various components
            EncryptionParameters parms = new EncryptionParameters(SchemeType.CKKS);
            parms.PolyModulusDegree = 8192;
            parms.CoeffModulus = DefaultParams.CoeffModulus128(polyModulusDegree: 8192);
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

            // This is the 'evaluator' section
            // this is not a part of the client and would, in a cloud based solution, be run on the server
            // the server should not know the values of the input features, but it will do math on them
            // likewise, the client would not know the various weights and parameters, but will receive a result
            double[] weights_model =
                    {0.0921533,
                    0.14545279,
                    0.11066622,
                    0.06119513,
                    0.10041948,
                    0.19091597,
                    0.07359407,
                    0.04503237,
                    0.10848583,
                    0.10092494};
            double intercept_model = -2.484390163425502;
            double[] weights_groundTruth =
                { 0.1,0.15,0.1,0.05,0.1,0.2,0.05,0.05,0.1,0.1};

            // we need to encode the weights/parameters/intercepts/etc. used by the model using the same public key as the client
            List<Ciphertext> weightsCTs = new List<Ciphertext>();
            List<Ciphertext> scoreCTs = new List<Ciphertext>();
            foreach (var weight in weights_model)
            {
                List<double> weightVector = new double[] { weight }.ToList();
                List<double> scoreVector = new double[] { 0.0 }.ToList();
                Plaintext weightPT = new Plaintext();
                encoder.Encode(weight, scale, weightPT);
                Ciphertext weightCT = new Ciphertext();
                encryptor.Encrypt(weightPT, weightCT);
                weightsCTs.Add(weightCT);
            }

            // next, we run the actual model's computation
            // linear regression is a basic dot product
            for (int i = 0; i < featureCiphers.Count(); i++)
            {
                List<Ciphertext> multResultCTs = new List<Ciphertext>();
                for (var j = 0; j < weightsCTs.Count; j++)
                {
                    Ciphertext multResultCT = new Ciphertext();
                    evaluator.Multiply(weightsCTs[j], featureCiphers[i][j], multResultCT);
                    multResultCTs.Add(multResultCT);
                }
                Ciphertext dotProductResult = new Ciphertext();
                evaluator.AddMany(multResultCTs, dotProductResult);

                Plaintext scorePT = new Plaintext();
                encoder.Encode(intercept_model, dotProductResult.Scale, scorePT);
                Ciphertext scoreCT = new Ciphertext();
                encryptor.Encrypt(scorePT, scoreCT);
                evaluator.AddInplace(scoreCT, dotProductResult);
                scoreCTs.Add(scoreCT);
                //evaluator.AddInplace(scoreCTs[i], interceptCT);
            }

            // we now have the encrypted version of the ML model. The below section is the client's again
            // in it, we decrypt the results given by the server using the private key generated previously
            List<List<double>> predictions = new List<List<double>>();
            for(int i = 0; i < scoreCTs.Count; i++)
            {
                Plaintext plainResult = new Plaintext();
                var encryptedResult = scoreCTs[i];
                decryptor.Decrypt(encryptedResult, plainResult);
                List<double> result = new List<double>();
                encoder.Decode(plainResult, result);
                predictions.Add(result);
            }

            // this is the output section. In practice, we would not have access to these values (as they represent the ground truth)
            // We are using them here merely to demonstrate that we can properly recover the proper ML model output
            double[] yGroundTruth = { 68.43881952, 87.75905253, 88.34053641, 83.87264322, 95.20322583, 89.61704108 };
            double[] yTest = { 68.702934, 87.28860458, 88.40827187, 83.24001674, 94.53137951, 89.00229455 };
            const double EPSILON = 1e-4;
            for (int i = 0; i < predictions.Count; i++)
            {
                var avgDecryption = predictions[i].Average();
                var absDelta = Math.Abs(avgDecryption - yTest[i]);
                Assert.IsTrue(absDelta < EPSILON);
            }
        }
    }
}
