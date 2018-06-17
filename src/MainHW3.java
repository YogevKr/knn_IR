import weka.core.Instances;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;

public class MainHW3 {

    public static BufferedReader readDataFile(String filename) {
        BufferedReader inputReader = null;

        try {
            inputReader = new BufferedReader(new FileReader(filename));
        } catch (FileNotFoundException ex) {
            System.err.println("File not found: " + filename);
        }

        return inputReader;
    }

    public static Instances loadData(String fileName) throws IOException {
        BufferedReader datafile = readDataFile(fileName);
        Instances data = new Instances(datafile);
        data.setClassIndex(data.numAttributes() - 1);
        return data;
    }

    public static void main(String[] args) throws Exception {
        Instances trainingAutoPrice = loadData("/Users/yogev/Google Drive/IDC/Year 2/Semester 2/Machine Learning from Data/HW/3/HomeWork3/src/HomeWork3/auto_price.txt");
        trainingAutoPrice.randomize(new Random(1));

        Instances scaled_trainingAutoPrice = FeatureScaler.scaleData(trainingAutoPrice);


        Knn knn = new Knn();
        knn.buildClassifier(trainingAutoPrice);

        Knn.WeightingScheme chosenWeightingScheme = null;
        Knn.LpDistance chosenP = null;
        int chosenK = 0;
        double bestError = Double.MAX_VALUE, error;

        // Original data
        for (Knn.WeightingScheme weightingScheme : Knn.WeightingScheme.values()) {
            for (Knn.LpDistance p : Knn.LpDistance.values()) {
                for (int k = 1; k <= 20; k++) {
                    knn.setUp(weightingScheme, p, Knn.DistanceCheck.Regular, k);
                    error = knn.crossValidationError(trainingAutoPrice, 10);

                    if (error < bestError) {
                        chosenWeightingScheme = weightingScheme;
                        chosenK = k;
                        chosenP = p;
                        bestError = error;
                    }
                }
            }
        }


        Knn scaled_knn = new Knn();
        scaled_knn.buildClassifier(scaled_trainingAutoPrice);

        Knn.WeightingScheme scaled_chosenWeightingScheme = null;
        Knn.LpDistance scaled_chosenP = null;
        int scaled_chosenK = 0;
        double scaled_bestError = Double.MAX_VALUE, scaled_error;

        // scaled
        for (Knn.WeightingScheme weightingScheme : Knn.WeightingScheme.values()) {
            for (Knn.LpDistance scaled_p : Knn.LpDistance.values()) {
                for (int scaled_k = 1; scaled_k <= 20; scaled_k++) {
                    scaled_knn.setUp(weightingScheme, scaled_p, Knn.DistanceCheck.Regular, scaled_k);
                    scaled_error = scaled_knn.crossValidationError(scaled_trainingAutoPrice, 10);

                    if (scaled_error < scaled_bestError) {
                        scaled_chosenWeightingScheme = weightingScheme;
                        scaled_chosenK = scaled_k;
                        scaled_chosenP = scaled_p;
                        scaled_bestError = scaled_error;
                    }
                }
            }
        }

        System.out.println("----------------------------");
        System.out.println("Results for original dataset: ");
        System.out.println("----------------------------");

        System.out.println("Cross validation error with K =  " + chosenK + ", lp = " + chosenP +
                ", majority function = " + chosenWeightingScheme + " for auto_price data is: " + bestError);

        System.out.println();

        System.out.println("----------------------------");
        System.out.println("Results for scaled dataset: ");
        System.out.println("----------------------------");

        System.out.println("Cross validation error with K =  " + scaled_chosenK + ", lp = " + scaled_chosenP +
                ", majority function = " + scaled_chosenWeightingScheme + " for auto_price data is: " + scaled_bestError);


        int[] fold = {trainingAutoPrice.numInstances(), 50, 10, 3};
        for (int foldNumber : fold) {
            System.out.println();
            System.out.println("----------------------------");
            System.out.println("Results for " + foldNumber + " folds: ");
            System.out.println("----------------------------");

            for (Knn.DistanceCheck distanceCheck : Knn.DistanceCheck.values()) {
                scaled_knn.setUp(scaled_chosenWeightingScheme, scaled_chosenP, distanceCheck, scaled_chosenK);
                error = scaled_knn.crossValidationError(trainingAutoPrice, foldNumber);

                System.out.print("Cross validation error of " + distanceCheck + " knn on auto_price dataset is " + error + " and ");
                System.out.println("the average elapsed time is " + scaled_knn.getAverageCVRunningTime());
                System.out.println("The total elapsed time is: " + scaled_knn.getTotalCVRunningTime());
                System.out.println();
            }
        }
    }
}
