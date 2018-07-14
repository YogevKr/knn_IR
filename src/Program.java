import org.apache.lucene.queryparser.classic.ParseException;

import java.io.*;
import java.util.*;


public class Program {

    private static final int NUM_OF_STOP_WORDS = 20;

    private static String m_TestFile;
    private static String m_TrainFile;
    private static File m_OutputFile;

    private static String m_WorkingDir;
    private static int m_K;


    public static void main(String[] args) throws IOException {

        if (args.length != 1) {
            System.out.println("Software except exactly one parameter");
            System.exit(1);
        }
        initFromParameterFile(args[0]);

        // Choose Data Set:
        String dataSet = m_TrainFile;

        Knn tempEngine = new Knn();
        tempEngine.InitStopWords();
        tempEngine.SetAnalyzer();
        tempEngine.SetIndex();
        tempEngine.AddDocsFile(dataSet);

        Knn knn = new Knn();
        knn.SetRetrievalAlgorithm();
        knn.InitStopWords();
        try {
            knn.SetStopWords(tempEngine.GetMostCommonTerms(NUM_OF_STOP_WORDS));
        } catch (Exception e) {
            e.printStackTrace();
        }
        knn.SetAnalyzer();
        knn.SetIndex();
        knn.AddDocsFile(dataSet);

        knn.SetClassifier(m_K);
        ArrayList<String[]> testSetPrediction = knn.SetPrediction(m_TestFile);
        System.out.println(String.format("K = %d P = %f", m_K, calculatePrecise(testSetPrediction)));

        Utils.ListToCSV(m_OutputFile,testSetPrediction);
    }

    private static int findBestK(int i_min, int i_max, Knn i_knn, String i_TestFile) throws IOException {
        int bestK = 0;
        double bestP = 0, p;

        for (int i = i_min; i < i_max; i++) {
            i_knn.SetClassifier(m_K);
            ArrayList<String[]> testSetPrediction = i_knn.SetPrediction(i_TestFile);
            p = calculatePrecise(testSetPrediction);
            System.out.println(String.format("K = %d P = %f", i, p));

            if (p > bestP){
                bestK = i;
                bestP = p;
            }
        }
        return bestK;
    }

    private static double calculatePrecise(ArrayList<String[]> i_TestSetPrediction){
        int falsePredictions = 0;

        for (String[] doc:i_TestSetPrediction) {
            if (!doc[1].equals(doc[2])){
                falsePredictions++;
            }
        }

        return 1 - ((falsePredictions * 1.0) / i_TestSetPrediction.size());
    }

    private static void initFromParameterFile(String i_parameterFilePath) {
        ArrayList<String> lines = Utils.fileToLineList(i_parameterFilePath);

        for (String line : lines) {

//            trainFile=train.csv
//            testFile=test.csv
//            outputFile=out/out1.csv
//            k=20

            if (line.startsWith("trainFile=")) {
                m_TrainFile = line.substring(line.indexOf('=') + 1);
            } else if ((line.startsWith("testFile="))) {
                m_TestFile = line.substring(line.indexOf('=') + 1);
            } else if ((line.startsWith("outputFile="))) {
                m_OutputFile = new File(line.substring(line.indexOf('=') + 1));
            } else if ((line.startsWith("k="))) {
                m_K = Integer.parseInt(line.substring(line.indexOf('=') + 1).replaceAll("[^\\d.]", ""));
            }
        }
    }
}