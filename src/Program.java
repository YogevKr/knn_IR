import org.apache.lucene.queryparser.classic.ParseException;

import java.io.*;
import java.util.*;


public class Program {

    private static final int NUM_OF_STOP_WORDS = 20;
    private static final double ALPHA = 0.5;
    private static final double BETA = 1;

    private static FileReader m_TestFile;
    private static FileReader m_TrainFile;
    private static FileWriter m_OutputFile;

    private static String m_WorkingDir;
    private static String m_K = "";
    private static Knn m_SearchEngine;
    private static Map<Integer, Map<String, Float>> m_QueriesResults;
    private static Map<Integer, String[]> m_Truth;


    public static void main(String[] args) throws IOException, ParseException {

        if (args.length != 1) {
            System.out.println("Software except exactly one parameter");
            System.exit(1);
        }

        initFromParameterFile(args[0]);

        Knn tempEngine = new Knn();
        tempEngine.InitStopWords();
        tempEngine.SetAnalyzer();
        tempEngine.SetIndex();
        tempEngine.AddDocsFile(m_TrainFile);

        m_SearchEngine = new Knn();
        m_SearchEngine.SetRetrievalAlgorithm();
        m_SearchEngine.InitStopWords();
        try {
            m_SearchEngine.SetStopWords(tempEngine.GetMostCommonTerms(NUM_OF_STOP_WORDS));
        } catch (Exception e) {
            e.printStackTrace();
        }
        m_SearchEngine.SetAnalyzer();
        m_SearchEngine.SetIndex();

        //// Find Best T

//        findBestTForGivenTruth("truth.txt");

//        IndexReader reader = DirectoryReader.open(m_SearchEngine.m_Index);
//        StandardAnalyzer standardAnalyzer = new StandardAnalyzer(StopFilter.makeStopSet(m_SearchEngine.m_StopWordList));
//
//        KNearestNeighborClassifier knn = new KNearestNeighborClassifier(reader, m_SearchEngine.m_SimilarityMethod, standardAnalyzer, null, 5, MoreLikeThis.DEFAULT_MIN_DOC_FREQ, MoreLikeThis.DEFAULT_MIN_TERM_FREQ, "docId", "content");
//
//        List<ClassificationResult<BytesRef>> test = knn.getClasses("Test");


//        parseTheTruth("truth.txt");
//        runExperiment();

    }

    private static double runExperiment() {
        double sumOfF = 0, sumOfP = 0, sumOfR = 0;

        for (int i = 1; i <= m_Truth.size(); i++) {

            //TP
            ArrayList<String> predictionDocs = new ArrayList<>(m_QueriesResults.get(i).keySet());
            predictionDocs.retainAll(new ArrayList<>(Arrays.asList(m_Truth.get(i))));
            int TP = predictionDocs.size();

            //FP
            predictionDocs = new ArrayList<>(m_QueriesResults.get(i).keySet());
            predictionDocs.removeAll(Arrays.asList(m_Truth.get(i)));
            int FP = predictionDocs.size();

            //FN
            predictionDocs = new ArrayList<>(m_QueriesResults.get(i).keySet());
            ArrayList<String> truth = new ArrayList<>(Arrays.asList(m_Truth.get(i)));
            truth.removeAll(predictionDocs);
            int FN = truth.size();

            double precision = (TP + FP != 0) ? TP / (TP + FP + 0.0) : 0;
            double recall = TP / (TP + FN + 0.0);

            sumOfP += precision;
            sumOfR += recall;

            double F = 1 / ((ALPHA / precision) + ((1 - ALPHA) / recall));

            sumOfF += F;
            System.out.println(String.format("%d Precision = %.2f Recall = %.2f F = %.2f", i, precision, recall, F));
        }

        System.out.println(String.format("Average Precision = %f Average recall = %f Average F = %f", sumOfP / m_Truth.size(), sumOfR / m_Truth.size(), sumOfF / m_Truth.size()));

        return sumOfF / m_Truth.size();
    }


    private static void initFromParameterFile(String i_parameterFilePath) throws FileNotFoundException {
        ArrayList<String> lines = Utils.fileToLineList(i_parameterFilePath);

        for (String line : lines) {

//            trainFile=train.csv
//            testFile=test.csv
//            outputFile=out/out1.csv
//            k=20


            if (line.startsWith("trainFile=")) {
                m_TrainFile = new FileReader(line.substring(line.indexOf('=') + 1));
            } else if ((line.startsWith("testFile="))) {
                m_TestFile = new FileReader(line.substring(line.indexOf('=') + 1));
            } else if ((line.startsWith("outputFile="))) {
                try {
                    m_OutputFile = new FileWriter(line.substring(line.indexOf('=') + 1));
                } catch (IOException e) {
                    e.printStackTrace();
                    System.exit(1);
                }
            } else if ((line.startsWith("k="))) {
                m_K = line.substring(line.indexOf('=') + 1);
            }
        }
    }
}