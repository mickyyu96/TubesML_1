import java.io.BufferedReader;
import java.io.FileReader;

import weka.classifiers.Evaluation;
import weka.core.Instances;

public class mainID3 {	
	public static void main(String[] args) throws Exception {
		String filename = "D:\\weather.nominal.arff";
		filename = "D:\\soybean.arff";
		BufferedReader reader = new BufferedReader(new FileReader(filename));
		Instances data = new Instances(reader);
		data.setClassIndex(data.numAttributes() - 1);
		reader.close();
		
		myID3 id3 = new myID3();
		id3.buildClassifier(data);
		
		Evaluation eval = new Evaluation(data);
		eval.evaluateModel(id3, data);
		
		System.out.println();
		System.out.println("=== Summary ===");
		System.out.println(eval.toSummaryString());
		System.out.println(eval.toMatrixString());
	}
}
