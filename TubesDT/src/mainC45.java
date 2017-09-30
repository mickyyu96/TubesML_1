import java.io.BufferedReader;
import java.io.FileReader;

import weka.classifiers.Evaluation;
import weka.core.Instances;

public class mainC45 {	
	public static void main(String[] args) throws Exception {
		String filename = "/Users/atikazzahra/Documents/Atikazzahra/ProgrammingRelated/TubesML_1/motor.arff";
		//filename = "D:\\soybean.arff";
		BufferedReader reader = new BufferedReader(new FileReader(filename));
		Instances data = new Instances(reader);
		data.setClassIndex(data.numAttributes() - 1);
		reader.close();
		
		System.out.println("C45");
		
		myC45RP c45 = new myC45RP();
		c45.buildClassifier(data);
		
		Evaluation eval = new Evaluation(data);
		eval.evaluateModel(c45, data);
		
		System.out.println();
		System.out.println("=== Summary ===");
		System.out.println(eval.toSummaryString());
		System.out.println(eval.toMatrixString());
	}
}
