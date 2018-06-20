import org.apache.lucene.analysis.StopFilter;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.classification.ClassificationResult;
import org.apache.lucene.classification.KNearestNeighborClassifier;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.misc.HighFreqTerms;
import org.apache.lucene.misc.TermStats;
import org.apache.lucene.queryparser.classic.ParseException;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.*;
import org.apache.lucene.search.similarities.*;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.RAMDirectory;
import org.apache.lucene.util.BytesRef;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;


public class Knn {

    public static final int DOC_ID_I = 0;
    public static final int LABEL_I = 1;
    public static final int TITLE_I = 2;
    public static final int TEXT_I = 3;

    public static final String DOC_ID = "DOC_ID";
    public static final String LABEL = "LABEL";
    public static final String TITLE = "TITLE";
    public static final String TEXT = "TEXT";

    private int m_trainingSetPercent = 20;
    private ArrayList<String> m_StopWordList;
    private StandardAnalyzer m_Analyzer;
    private Directory m_Index;
    private IndexWriterConfig m_IndexWriterConfig;
    private Similarity m_SimilarityMethod;
    private KNearestNeighborClassifier m_Classifier;

    public void SetClassifier(int i_K) throws IOException {
        IndexReader reader = DirectoryReader.open(m_Index);

        m_Classifier = new KNearestNeighborClassifier(reader, m_SimilarityMethod, m_Analyzer, null, i_K,
                1, 1, LABEL, TITLE, TEXT);
    }

    public String Prediction(String[] i_Data) throws IOException {
        return m_Classifier.getClasses(i_Data[TEXT_I]).get(0).getAssignedClass().utf8ToString();
    }

    public ArrayList<String[]> SetPrediction(String i_DataSet) throws IOException {
        ArrayList<String[]> docsFileLines = Utils.ReadCsvFile(i_DataSet);
        ArrayList<String[]> results = new ArrayList<>();
        Random rand = new Random();

        for (String[] doc : docsFileLines){
            results.add(new String[]{doc[DOC_ID_I], Prediction(doc), doc[LABEL_I]});
        }

        return results;
    }

    public void AddDocsFile(String i_DocsFile) throws IOException {

        ArrayList<String[]> docsFileLines = Utils.ReadCsvFile(i_DocsFile);
        IndexWriter w = new IndexWriter(m_Index, m_IndexWriterConfig);
        Random rand = new Random();

        for (String[] doc : docsFileLines) {
            int n = rand.nextInt(100) + 1;
            if (n < m_trainingSetPercent){
                addDoc(w, doc);
            }
        }

        w.close();
    }

    public void InitStopWords() {
        m_StopWordList = new ArrayList<>();
    }

    public void SetStopWords(ArrayList<String> i_termList) {
        m_StopWordList = i_termList;
    }

    public ArrayList<String> GetMostCommonTerms(int i_n) throws Exception {
        ArrayList<String> termList = new ArrayList<>();
        IndexReader reader = DirectoryReader.open(m_Index);
        TermStats[] terms = HighFreqTerms.getHighFreqTerms(reader, i_n,
                TEXT, new HighFreqTerms.DocFreqComparator());

        for (TermStats term : terms) {
            termList.add(term.termtext.utf8ToString());
        }

        reader.close();

        return termList;
    }


    public void SetAnalyzer() {
        //    Specify the analyzer for tokenizing text.
        //    The same analyzer should be used for indexing and searching

        m_Analyzer = new StandardAnalyzer(StopFilter.makeStopSet(m_StopWordList));
    }

    public void SetIndex() {
        m_Index = new RAMDirectory();
        m_IndexWriterConfig = new IndexWriterConfig(m_Analyzer);
    }

    public void SetRetrievalAlgorithm() {

        m_SimilarityMethod = new SimilarityBase() {
            @Override
            protected float score(BasicStats i_basicStats, float i_tf, float i_docLen) {
                long N = i_basicStats.getNumberOfDocuments();
                double DFt = i_basicStats.getDocFreq() + 1;

                double idf = Math.log10(N / DFt);

                return (float) (1 + Math.log10(i_tf) * idf);
            }

            @Override
            public String toString() {
                return "TF-IDF";
            }
        };
    }

    private void addDoc(IndexWriter i_w, String[] i_data) throws IOException {
        Document doc = new Document();

        doc.add(new StringField(DOC_ID, i_data[DOC_ID_I], Field.Store.YES));
        doc.add(new StringField(LABEL, i_data[LABEL_I], Field.Store.YES));
        doc.add(new TextField(TITLE, i_data[TITLE_I], Field.Store.YES));
        doc.add(new TextField(TEXT, i_data[TEXT_I], Field.Store.YES));

        i_w.addDocument(doc);
    }


}
