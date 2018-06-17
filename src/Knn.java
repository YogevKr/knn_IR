import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

import java.util.PriorityQueue;

class DistanceCalculator {

    public static double distance(Instance one, Instance two, int p) {
        return lpDistance(one, two, p);
    }

    public static double distance(Instance one, Instance two, double threshold, int p) {
        return efficientLpDistance(one, two, threshold, p);
    }

    public static double distance(Instance one, Instance two) {
        return lInfinityDistance(one, two);
    }

    public static double distance(Instance one, Instance two, double threshold) {
        return efficientLInfinityDistance(one, two, threshold);
    }

    /**
     * Returns the Lp distance between 2 instances.
     *
     * @param one
     * @param two
     */
    private static double lpDistance(Instance one, Instance two, int p) {
        int numOfAttributes = one.numAttributes() - 1;
        double absoluteValue, powerOfDifference, sum = 0;

        for (int i = 0; i < numOfAttributes; i++) {
            powerOfDifference = Math.pow((one.value(i) - two.value(i)), p);
            absoluteValue = Math.abs(powerOfDifference);
            sum += absoluteValue;
        }

        return Math.pow(sum, (1 / p));
    }

    /**
     * Returns the L infinity distance between 2 instances.
     *
     * @param one
     * @param two
     * @return
     */
    private static double lInfinityDistance(Instance one, Instance two) {
        int numOfAttributes = one.numAttributes() - 1;
        double max = 0, different;

        for (int i = 0; i < numOfAttributes; i++) {
            different = Math.abs(one.value(i) - two.value(i));

            if (different > max) {
                max = different;
            }
        }
        return max;
    }

    /**
     * Returns the Lp distance between 2 instances, while using an efficient distance check.
     *
     * @param one
     * @param two
     * @return
     */
    private static double efficientLpDistance(Instance one, Instance two, double threshold, int p) {
        int numOfAttributes = one.numAttributes() - 1;
        double absoluteValue, powerOfDifference, sum = 0;


        for (int i = 0; i < numOfAttributes; i++) {

            powerOfDifference = Math.pow((one.value(i) - two.value(i)), p);
            absoluteValue = Math.abs(powerOfDifference);
            sum += absoluteValue;

            if (sum > threshold) {
                sum = Double.MAX_VALUE;
                break;
            }
        }

        if (sum != Double.MAX_VALUE) {
            return sum;
        } else {
            return Math.pow(sum, (1 / p));
        }
    }

    /**
     * Returns the Lp distance between 2 instances, while using an efficient distance check.
     *
     * @param one
     * @param two
     * @return
     */
    private static double efficientLInfinityDistance(Instance one, Instance two, double threshold) {
        int numOfAttributes = one.numAttributes() - 1;
        double max = 0, different;

        for (int i = 0; i < numOfAttributes; i++) {
            different = Math.abs(one.value(i) - two.value(i));

            if (different > max) {
                max = different;

                if (max > threshold) {
                    max = Double.MAX_VALUE;
                }
            }
        }
        return max;
    }
}

class Entry implements Comparable<Entry> {
    private Instance instance;
    private double distance;

    public Entry(Instance instance, double distance) {
        this.instance = instance;
        this.distance = distance;
    }

    public Instance getInstance() {
        return this.instance;
    }

    public double getDistance() {
        return this.distance;
    }

    @Override
    public int compareTo(Entry other) {
        return Double.compare(other.getDistance(), this.getDistance());
    }
}

public class Knn implements Classifier {

    public enum WeightingScheme {Uniform, Weighted}

    public enum LpDistance {
        One(1), Two(2), Three(3), Infinity(0);
        private int p;

        LpDistance(int p) {
            this.p = p;
        }

        public int getP() {
            return p;
        }
    }

    public enum DistanceCheck {Regular, Efficient}

    private Instances m_TrainingInstances;
    private Instances m_TrainingInstances_Backup;


    private WeightingScheme m_WeightingScheme;
    private LpDistance m_P;
    private boolean m_EfficientCheck;
    private int m_K;

    private long m_AverageCVRunningTime;
    private long m_TotalCVRunningTime;


    @Override
    /**
     * Build the knn classifier. In our case, simply stores the given instances for 
     * later use in the prediction.
     * @param instances
     */
    public void buildClassifier(Instances instances) throws Exception {
        m_TrainingInstances = instances;
    }

    public void setUp(WeightingScheme i_WeightingScheme, LpDistance i_p,
                      DistanceCheck i_DistanceCheck, int k) {

        m_WeightingScheme = i_WeightingScheme;
        m_P = i_p;
        m_EfficientCheck = i_DistanceCheck == DistanceCheck.Efficient;
        m_K = k;

    }

    private double distance(Instance one, Instance two) {
        // Non efficient, p=INF
        if (m_P == LpDistance.Infinity) {
            return DistanceCalculator.distance(one, two);
        }
        // Non efficient, p!=INF
        else {
            return DistanceCalculator.distance(one, two, m_P.getP());
        }
    }

    private double distance(Instance one, Instance two, double threshold) {
        // Efficient, p=INF
        if (m_P == LpDistance.Infinity) {
            return DistanceCalculator.distance(one, two, threshold);
        }
        // Efficient, p!=INF
        else {
            return DistanceCalculator.distance(one, two, threshold, m_P.getP());
        }
    }

    /**
     * Returns the knn prediction on the given instance.
     *
     * @param instance
     * @return The instance predicted value.
     */
    public double regressionPrediction(Instance instance) {
        PriorityQueue<Entry> heap = findNearestNeighbors(instance);

        if (m_WeightingScheme == WeightingScheme.Weighted) {
            return getWeightedAverageValue(heap);
        } else {
            return getAverageValue(heap);
        }
    }

    /**
     * Caclcualtes the average error on a give set of instances.
     * The average error is the average absolute error between the target value and the predicted
     * value across all insatnces.
     *
     * @param instances
     * @return
     */
    public double calcAvgError(Instances instances) {
        double sumOfErrors = 0, prediction;
        Instance currentInstance;

        for (int i = 0; i < instances.numInstances(); i++) {
            currentInstance = instances.get(i);
            prediction = regressionPrediction(currentInstance);

            sumOfErrors += Math.abs(prediction - currentInstance.classValue());
        }
        return sumOfErrors / instances.numInstances();
    }

    /**
     * Calculates the cross validation error, the average error on all folds.
     *
     * @param instances    Insances used for the cross validation
     * @param num_of_folds The number of folds to use.
     * @return The cross validation error.
     */
    public double crossValidationError(Instances instances, int num_of_folds) throws Exception {
        m_TrainingInstances_Backup = m_TrainingInstances;

        Instances trainingSet, validationSet;

        double errorSum = 0;

        long startTime;
        long estimatedTime;
        long sumOfEstimatedTime = 0;

        for (int i = 0; i < num_of_folds; i++) {

            validationSet = instances.testCV(num_of_folds, i);
            trainingSet = instances.trainCV(num_of_folds, i);
            m_TrainingInstances = trainingSet;

            startTime = System.nanoTime();
            errorSum += calcAvgError(validationSet);
            estimatedTime = System.nanoTime() - startTime;

            sumOfEstimatedTime += estimatedTime;
        }

        m_TotalCVRunningTime = sumOfEstimatedTime;
        m_AverageCVRunningTime = sumOfEstimatedTime / num_of_folds;

        m_TrainingInstances = m_TrainingInstances_Backup;
        return errorSum / num_of_folds;
    }

    public long getAverageCVRunningTime() {
        return m_AverageCVRunningTime;
    }

    public long getTotalCVRunningTime() {
        return m_TotalCVRunningTime;
    }

    /**
     * Finds the k nearest neighbors.
     *
     * @param instance
     */
    public PriorityQueue<Entry> findNearestNeighbors(Instance instance) {
        PriorityQueue<Entry> heap = new PriorityQueue<>();

        if (!m_EfficientCheck) {
            for (int i = 0; i < m_TrainingInstances.numInstances(); i++) {
                Instance currentInstance = m_TrainingInstances.get(i);
//                if (!currentInstance.equals(instance)) {
                    heap.add(new Entry(currentInstance, distance(currentInstance, instance)));
                    if (heap.size() > m_K) {
                        heap.poll();
                    }
//                }
            }
        } else {
            for (int i = 0; i < m_TrainingInstances.numInstances(); i++) {
                Instance currentInstance = m_TrainingInstances.get(i);
                if (!currentInstance.equals(instance))
                    if (!heap.isEmpty()) {
                        heap.add(new Entry(currentInstance, distance(currentInstance, instance, heap.peek().getDistance())));
                        if (heap.size() > m_K) {
                            heap.poll();
                        }
                    } else {
                        heap.add(new Entry(currentInstance, distance(currentInstance, instance)));
                        if (heap.size() > m_K) {
                            heap.poll();
                        }
                    }
            }
        }
        return heap;
    }

    /**
     * Cacluates the average value of the given elements in the collection.
     *
     * @param
     * @return
     */
    public double getAverageValue(PriorityQueue<Entry> heap) {
        double sum = 0;
        Entry entry;

        while (!heap.isEmpty()) {
            entry = heap.poll();
            sum += entry.getInstance().classValue();
        }
        return sum / m_K;
    }

    /**
     * Calculates the weighted average of the target values of all the elements in the collection
     * with respect to their distance from a specific instance.
     *
     * @return
     */
    public double getWeightedAverageValue(PriorityQueue<Entry> heap) {
        double sumOfOneOverSquaredDistance = 0, sumOfValueDividedSquaredDistance = 0;
        double distanceSquared, instanceValue;
        Entry currentEntry;

        while (!heap.isEmpty()) {
            currentEntry = heap.poll();

            distanceSquared = Math.pow(currentEntry.getDistance(), 2);
            instanceValue = currentEntry.getInstance().classValue();

            if (distanceSquared != 0) {
                sumOfValueDividedSquaredDistance += instanceValue / distanceSquared;
                sumOfOneOverSquaredDistance += 1 / distanceSquared;
            }
        }
        return sumOfValueDividedSquaredDistance / sumOfOneOverSquaredDistance;
    }


    @Override
    public double[] distributionForInstance(Instance arg0) throws Exception {
        // TODO Auto-generated method stub - You can ignore.
        return null;
    }

    @Override
    public Capabilities getCapabilities() {
        // TODO Auto-generated method stub - You can ignore.
        return null;
    }

    @Override
    public double classifyInstance(Instance instance) {
        // TODO Auto-generated method stub - You can ignore.
        return 0.0;
    }
}
