import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.trees.j48.ClassifierSplitModel;

import java.util.Enumeration;

public class splitC45 {

	private double m_perClassPerBag[][];
	private double m_perBag[];
	private double m_perClass[];
	private double m_splitPoint = Double.MAX_VALUE;
	private double m_infoGain = 0;
	private int m_index = 0;
	private Instances m_leftInstances;
	private Instances m_rightInstances;
	
	public splitC45() {}
	
	public double perClassPerBag(int bagIndex, int classIndex) {
		return m_perClassPerBag[classIndex][bagIndex];
	}
	
	public double perBag(int bagIndex) {
		return m_perBag[bagIndex];
	}
	
	public double perClass(int classIndex) {
		return m_perClass[classIndex];
	}
	
	public double[] getPerBag() {
		return m_perBag;
	}
	
	public double splitPoint() {
		return m_splitPoint;
	}
	
	public double infoGain() {
		return m_infoGain;
	}
	
	public Instances leftInstances() {
		return m_leftInstances;
	}
	
	public Instances rightInstances() {
		return m_rightInstances;
	}
	
	public boolean isSplit() {
		return m_index != 0;
	}
	
	public void handleNumericAttribute(int m_attIndex, Instances data) throws Exception {
		int splitIndex = -1;
		int i;
		int classIndex;
		int firstMiss;
		int next;
		int last;
		int numBags = 2;
		int numClasses = data.numClasses();
		int minSplit = 2;
		double probLeft, probRight, prob;
		double entropyLeft, entropyRight, entropy;
		double currentInfoGain;
		double total;
		Instance instance;
		
		m_perClassPerBag = new double[numBags][0];
		m_perBag = new double[numBags];
		m_perClass = new double[numClasses];
		for (i = 0; i < numBags; i++) {
			m_perClassPerBag[i] = new double[numClasses];
		}
		total = 0;
		
		// Only Instances with known values are relevant
		Enumeration<Instance> enu = data.enumerateInstances();
		i = 0;
		while (enu.hasMoreElements()) {
			instance = enu.nextElement();
			if(instance.isMissing(m_attIndex)) {
				break;
			}
			classIndex = (int) instance.classValue();
			m_perClassPerBag[1][classIndex]++;
			m_perBag[1]++;
			m_perClass[classIndex]++;
			total++;
		}
		firstMiss = i;
		
		next = 1;
		last = 0;
		while(next < firstMiss) {
			if (data.instance(next-1).value(m_attIndex) + 1e-5 < data.instance(next).value(m_attIndex)) {
				// Move class values for all Instances up to next possible split point
				for(i = last; i < next; i++) {
					instance = data.instance(i);
					classIndex = (int) instance.classValue();
					m_perClassPerBag[1][classIndex]--;
					m_perClassPerBag[0][classIndex]++;
					m_perBag[1]--;
					m_perBag[0]++;
				}
				
				if((m_perBag[0] >= minSplit) && (m_perBag[1] >= minSplit)) {
					entropy = 0;
					entropyLeft = 0;
					entropyRight = 0;
					probLeft = m_perBag[0]/total;
					probRight = m_perBag[1]/total;
					for(i = 0; i < numClasses; i++) {
						prob = (m_perClassPerBag[0][i] + m_perClassPerBag[1][i])/total;
				    	if (prob != 0.0) {
				    		entropy -= prob * Utils.log2(prob);
				    	}
					}
					for(i = 0 ; i < numClasses; i++) {
						prob = m_perClassPerBag[0][i]/m_perBag[0];
				    	if (prob != 0.0) {
				    		entropyLeft -= prob * Utils.log2(prob);
				    	}
					}
					for(i = 0 ; i < numClasses; i++) {
						prob = m_perClassPerBag[1][i]/m_perBag[1];
				    	if (prob != 0.0) {
				    		entropyRight -= prob * Utils.log2(prob);
				    	}
					}
					currentInfoGain = entropy - (probLeft * entropyLeft) + (probRight * entropyRight);
				
					if(currentInfoGain > m_infoGain) {
						m_infoGain = currentInfoGain;
						splitIndex = next - 1;
					}
					m_index++;
				}
				
				last = next;
			}
			next++;
		}
		
		// Check whether split candidate found
		if(m_index == 0) {
			return;
		}
		
		m_splitPoint = (data.instance(splitIndex+1).value(m_attIndex) + 
				data.instance(splitIndex).value(m_attIndex)) / 2;
		
		// Restore distribution for best split
		m_perClassPerBag = new double[numBags][0];
		m_perBag = new double[numBags];
		m_perClass = new double[numClasses];
		for (i = 0; i < numBags; i++) {
			m_perClassPerBag[i] = new double[numClasses];
		}
		total = 0;
		
		for(i = 0; i < splitIndex + 1; i++) {
			instance = data.instance(i);
			m_leftInstances.add(instance);
			classIndex = (int) instance.classValue();
			m_perClassPerBag[0][classIndex]++;
			m_perClass[classIndex]++;
			m_perBag[0]++;
			total++;
		}
		
		for(i = splitIndex + 1; i < firstMiss; i++) {
			instance = data.instance(i);
			m_rightInstances.add(instance);
			classIndex = (int) instance.classValue();
			m_perClassPerBag[1][classIndex]++;
			m_perClass[classIndex]++;
			m_perBag[1]++;
			total++;
		}
	}	
}
