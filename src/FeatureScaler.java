import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Standardize;

public class FeatureScaler {
	/**
	 * Returns a scaled version (using standarized normalization) of the given dataset.
	 * @param instances The original dataset.
	 * @return A scaled instances object.
	 */
	public static Instances scaleData(Instances instances) throws Exception {

		Standardize filter = new Standardize();
		Instances defaultStdData;

		filter.setInputFormat(instances);
		defaultStdData = Filter.useFilter(instances, filter);

		return defaultStdData;
	}
}