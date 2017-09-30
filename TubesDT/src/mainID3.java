import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.Random;
import java.util.Scanner;
import weka.filters.Filter;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.filters.supervised.attribute.Discretize;
import weka.filters.supervised.instance.Resample;
import weka.filters.unsupervised.attribute.Remove;

public class mainID3 {	
	public Instances ReadArff(String filename) throws Exception {
	    BufferedReader reader = new BufferedReader(
	                             new FileReader(filename));
	    Instances data = new Instances(reader);
	    data.setClassIndex(data.numAttributes() - 1);
	    reader.close();
	    
	    return data;
	}
	
	public Instances Resample(Instances data) throws Exception {
		Resample filter = new Resample();
		Instances newData;
		
		filter.setInputFormat(data);
		newData = Filter.useFilter(data, filter);
		
		return newData;
	}
	
	public Instances RemoveAttribute(Instances data, int idx) throws Exception {
		String[] options = new String[2];
        options[0] = "-R";
        options[1] = Integer.toString(idx);
        Remove remove = new Remove();
        remove.setOptions(options);
        remove.setInputFormat(data);
        Instances newData = Filter.useFilter(data, remove);
        
        return newData;
	}
	
	public Classifier TenFoldsCrossValidation(Instances data, Classifier cls) throws Exception{
		cls.buildClassifier(data);
		
		Evaluation eval = new Evaluation(data);
		eval.crossValidateModel(cls, data, 10, new Random(1));
		
		System.out.println();
		System.out.println("=== Summary ===");
		System.out.println(eval.toSummaryString());
		System.out.println(eval.toMatrixString());
		
		return cls;
	}
	
	public Classifier SplitTest(Instances data, int percent, Classifier cls) throws Exception {
		data.randomize(new Random(1));
		int trainSize = (int) Math.round(data.numInstances() * percent / 100);
		int testSize = data.numInstances() - trainSize;
		Instances train = new Instances(data, 0, trainSize);
		Instances test = new Instances(data, trainSize, testSize);
		
		cls.buildClassifier(train);
		Evaluation eval = new Evaluation(test);
		eval.evaluateModel(cls, test);
		
		System.out.println();
		System.out.println("=== Summary ===");
		System.out.println(eval.toSummaryString());
		System.out.println(eval.toMatrixString());
		
		return cls;
	}
	
	public Classifier FullTrainingSchema(Instances data, Classifier cls) throws Exception{
		cls.buildClassifier(data);
		
		Evaluation eval = new Evaluation(data);
		eval.evaluateModel(cls, data);
		
		System.out.println();
		System.out.println("=== Summary ===");
		System.out.println(eval.toSummaryString());
		System.out.println(eval.toMatrixString());
		
		return cls;
	}
	
	public void classifyData(Classifier model, Instances data) throws Exception {
		Evaluation eval = new Evaluation(data);
		eval.evaluateModel(model, data);
		
		System.out.println();
		System.out.println("=== Summary ===");
		System.out.println(eval.toSummaryString());
		System.out.println(eval.toMatrixString());
	}
	
	public void saveModel(String filename, Classifier cls) throws Exception {
		ObjectOutputStream output = new ObjectOutputStream(new FileOutputStream(filename));
		output.writeObject(cls);
		output.flush();
		output.close();
	}
	
	public Classifier loadModel(String filename) throws Exception{
		ObjectInputStream fileinput = new ObjectInputStream(new FileInputStream(filename));
		Classifier cls = (Classifier) fileinput.readObject();
		fileinput.close();
		return cls;
	}
	
	public static void main(String[] args) throws Exception {
		mainID3 mainID3 = new mainID3();
		Scanner input = new Scanner(System.in);
		
		System.out.println("==============================");
		System.out.println("===       Tubes ML 1       ===");
		System.out.println("===  by: micky, kepi, ade  ===");
		System.out.println("==============================");
		System.out.println();
		
		int pilihan;
		do {
			System.out.println("Menu: 1. Mengolah dataset");
			System.out.println("      2. Membaca model dan mengklasifikasi Instances");
			System.out.println("      3. Exit");
			System.out.print("Masukkan pilihan: ");
			pilihan = input.nextInt();
			
			if (pilihan == 1) {
				System.out.print("Masukkan file dataset: ");
				String filename = input.next();

				System.out.println("\nMembaca " + filename + "...");
				Instances data = mainID3.ReadArff(filename);
				
			    System.out.println("\nHeader dataset:\n");
			    System.out.println(new Instances(data, 0));
			    
			    int pilihan2;
				do {
					System.out.println("Menu: 1. Melakukan filter Resample pada data");
					System.out.println("      2. Melakukan penghapusan attribute pada data");
				    System.out.println("      3. Melakukan pembelajaran dengan algoritma myID3 (10-fold cross validation)");
					System.out.println("      4. Melakukan pembelajaran dengan algoritma myID3 (split test-training)");
					System.out.println("      5. Melakukan pembelajaran dengan algoritma myID3 (full-training)");
				    System.out.println("      6. Melakukan pembelajaran dengan algoritma J48 (10-fold cross validation)");
					System.out.println("      7. Melakukan pembelajaran dengan algoritma J48 (split test-training)");
					System.out.println("      8. Melakukan pembelajaran dengan algoritma J48 (full-training)");
				    System.out.println("      9. Melakukan pembelajaran dengan algoritma myC45 (10-fold cross validation)");
					System.out.println("      10. Melakukan pembelajaran dengan algoritma myC45 (split test-training)");
					System.out.println("      11. Melakukan pembelajaran dengan algoritma myC45 (full-training)");
					System.out.println("      12. Melakukan pembelajaran dengan algoritma myC45EE (10-fold cross validation)");
					System.out.println("      13. Melakukan pembelajaran dengan algoritma myC45EE (split test-training)");
					System.out.println("      14. Melakukan pembelajaran dengan algoritma myC45EE (full-training)");
					System.out.println("      15. Back");
					System.out.print("Masukkan pilihan: ");
					pilihan2 = input.nextInt();
					
					Classifier cls = new myID3();
					if (pilihan2 == 1) {
						data = mainID3.Resample(data);
						System.out.println("\nHeader dataset setelah filter:\n");
					    System.out.println(new Instances(data, 0));
					}
					else if (pilihan2 == 2) {
						System.out.print("Masukkan indeks attribute yang ingin dihapus: ");
						int idx = input.nextInt();
						data = mainID3.RemoveAttribute(data, idx);
						System.out.println("\nHeader dataset setelah filter:\n");
					    System.out.println(new Instances(data, 0));
					}
					else if (pilihan2 == 3) {
						cls = new myID3();
						cls = mainID3.TenFoldsCrossValidation(data, cls);
					}
					else if (pilihan2 == 4) {
						cls = new myID3();
						System.out.print("Masukkan persentase split: ");
						int percent = input.nextInt();
						cls = mainID3.SplitTest(data, percent, cls);
					}
					else if (pilihan2 == 5) {
						cls = new myID3();
						cls = mainID3.FullTrainingSchema(data, cls);
					}
					else if (pilihan2 == 6) {
						J48 tree = new J48();
						cls = mainID3.TenFoldsCrossValidation(data, tree);
					}
					else if (pilihan2 == 7) {
						J48 tree = new J48();
						System.out.print("Masukkan persentase split: ");
						int percent = input.nextInt();
						cls = mainID3.SplitTest(data, percent, tree);
					}
					else if (pilihan2 == 8) {
						J48 tree = new J48();
						cls = mainID3.FullTrainingSchema(data, tree);
					}
					else if (pilihan2 == 9) {
						cls = new myC45();
						cls = mainID3.TenFoldsCrossValidation(data, cls);
					}
					else if (pilihan2 == 10) {
						cls = new myC45();
						System.out.print("Masukkan persentase split: ");
						int percent = input.nextInt();
						cls = mainID3.SplitTest(data, percent, cls);
					}
					else if (pilihan2 == 11) {
						cls = new myC45();
						cls = mainID3.FullTrainingSchema(data, cls);
					}
					else if (pilihan2 == 12) {
						cls = new myC45EE();
						cls = mainID3.TenFoldsCrossValidation(data, cls);
					}
					else if (pilihan2 == 13) {
						cls = new myC45EE();
						System.out.print("Masukkan persentase split: ");
						int percent = input.nextInt();
						cls = mainID3.SplitTest(data, percent, cls);
					}
					else if (pilihan2 == 14) {
						cls = new myC45EE();
						cls = mainID3.FullTrainingSchema(data, cls);
					}
					else if (pilihan2 == 15) {
						System.out.println();
					}
					if (pilihan2 == 8 || pilihan2 == 3 || pilihan2 == 4 || pilihan2 == 5 || pilihan2 == 6 || 
							pilihan2 == 7 || pilihan2 == 9 || pilihan2 == 10 || pilihan2 == 11 ||
							pilihan == 12 || pilihan == 13 || pilihan == 14){
						System.out.println("Save model pembelajaran? (y/n)");
						System.out.print("Masukkan pilihan: ");
						char answer = (char) System.in.read();
						if(answer == 'y'){
							System.out.print("Masukkan destinasi penyimpanan: ");
							filename = input.next();
							mainID3.saveModel(filename, cls);
							System.out.println("Model berhasil disimpan pada "+filename);
						}
						System.out.println();
					}
				} while (pilihan2 != 15);
			}
			else if (pilihan == 2) {
				System.out.print("Masukkan file model: ");
				String filename;
				filename = input.next();
				System.out.println("\nMembaca model...\n");
				Classifier model = mainID3.loadModel(filename);
			    System.out.println(model);
			    
			    System.out.print("Masukkan file dataset: ");
				String testFile = input.next();
				
				System.out.println("\nMembaca " + testFile + "...");
				Instances data = mainID3.ReadArff(testFile);
				
				data.setClassIndex(data.numAttributes() - 1);
				
				mainID3.classifyData(model, data);
			}
			else if (pilihan == 3) {
				input.close();
			}
		} while (pilihan != 3);
		return;
	}
}
