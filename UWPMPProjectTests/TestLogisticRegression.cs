using Microsoft.Research.SEAL;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;

namespace UWPMPProjectTests
{
    [TestClass]
    public class TestLogisticRegression
    {
        [TestMethod]
        public void TestLogisticRegressionBinaryClassification_NoEncryption()
        {
            // these are the feature sets we will be encrypting and getting
            // the ML model results on along with the expected truth values of the ML model

            double[][] testX = new double[6][];
            testX[0] = new double[] { 1, 1, 0, 0, 0, 1, 1, 0, 0, 1 };
            testX[1] = new double[] { 1, 0, 1, 0, 1, 1, 0, 1, 0, 1 };
            testX[2] = new double[] { 0, 1, 0, 0, 0, 1, 0, 1, 1, 0 };
            testX[3] = new double[] { 0, 1, 1, 0, 1, 1, 1, 1, 0, 0 };
            testX[4] = new double[] { 0, 1, 0, 1, 1, 1, 0, 0, 1, 1 };
            testX[5] = new double[] { 1, 0, 1, 1, 0, 1, 0, 0, 0, 1 };

            double[] testY = { 0, 0, 0, 1, 1, 1 };
            bool[] expectedModelResults = { false, false, true, true, true, true };

            // This is the 'evaluator' section
            // this is not a part of the client and would, in a cloud based solution, be run on the server
            // the server should not know the values of the input features, but it will do math on them
            double[] weights =
                    {-0.0448429813505995,
                     0.22603459546040847,
                     0.8256180461493858,
                     2.970324762165545,
                     -0.0022260364010428055,
                     -0.27601604216924047,
                     0.7643519117530984,
                     0.552635425157094,
                     -0.18622044388306305,
                     -2.2604158458243537};

            List<double> scores = new List<double>();
            for (int i = 0; i < testX.Length; i++)
            {
                double[] xFeatures = testX[i];
                double expectedY = testY[i];

                var score = 0.0;
                for (int j = 0; j < xFeatures.Length; j++)
                {
                    score += weights[j] * xFeatures[j];
                }
                score = 1.0 / (1.0 + Math.Exp(-1.0 * score));
                scores.Add(score);
            }
            List<bool> predictions = scores.Select(score => score > 0.5).ToList();

            for (int i = 0; i < predictions.Count; i++)
            {
                Assert.AreEqual(predictions[i], expectedModelResults[i]);
            }
        }
    }
}
