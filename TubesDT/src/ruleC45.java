import java.lang.Math;
import weka.core.Attribute;
import java.util.*;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;

import java.util.Enumeration;
import java.util.Random;

public class ruleC45{
	private HashMap<Attribute,Double> preconditions = new HashMap<Attribute, Double>();
	private double classValue;
	private double accuracy = 0;
	private double cErrorEstimate = 0.25;
	private double zErrorEstimate = 0.67;
	
	public ruleC45() {}
	public ruleC45(ruleC45 rule) {
		preconditions = new HashMap<Attribute, Double>(rule.preconditions);
		classValue = rule.classValue;
		accuracy = rule.accuracy;
	} 
	
	public void addPrecond(Attribute attr, double indexAttr) {
		preconditions.put(attr, indexAttr);
	}
	
	public void addClassValue(double val) {
		classValue = val;
	}
	
	public double getAccuracy() {
		return accuracy;
	}
	
	public double getClassValue() {
		return classValue;
	}
	
	public void printRule() {
		System.out.println("=====rule=====");
		for (Attribute key : preconditions.keySet()) {
			System.out.println("<"+key+" = "+preconditions.get(key)+">");
		}
		System.out.println("class value: "+classValue);
		System.out.println("accuracy: "+accuracy);
		System.out.println("==============");
	}
	
	public void prune(Instances test) {
		int maxacc = 0;
		Attribute maxacc_key = null;
		boolean pruned = true;
		accuracy = evaluate(test);
		if (preconditions.size()>1) {
			while (pruned) {
				Iterator<Attribute> it = preconditions.keySet().iterator();
				HashMap<Attribute,Double> newpreconds = new HashMap<Attribute,Double>();
				while (it.hasNext()) {
					Attribute key = it.next();
					Attribute lastkey = key;
					double lastval = preconditions.get(key);
					it.remove();
					double evalprecond = evaluate(test);
					//System.out.println("[ evalprecond: "+evalprecond+" ]>[ accuracy: "+accuracy+"]");
					if (evalprecond > accuracy) {
						accuracy = evalprecond;
						maxacc_key = lastkey;
					} 
					newpreconds.put(lastkey, lastval);
				}
				preconditions = new HashMap<Attribute,Double>(newpreconds);
				if (maxacc_key != null) {
					//printRule();
					preconditions.remove(maxacc_key);
					//System.out.println("-----pruned!----");
					//printRule();
					maxacc_key = null;
				} else {
					//printRule();
					pruned = false;
				}
			}
		}
	}
	
	public boolean classify(Instance data) {
		boolean valid = true;
		for (Attribute key : preconditions.keySet()) {
			if (data.value(key) != preconditions.get(key)) {
				valid = false;
			}
		}
		return valid;
	}
	
	private double getErrorEstimate(double f, double N) {
		double temp0 = (f/N) - (Math.pow(f, 2)*1.0/N) + (Math.pow(zErrorEstimate, 2)*1.0/(4.0*Math.pow(N, 2)));
		
		double temp1 = zErrorEstimate * Math.sqrt(temp0);
		double temp2 = Math.pow(zErrorEstimate, 2)/(2.0*N);
		double temp3 = temp1 + temp2 + f;
		double divider = 1 + (Math.pow(zErrorEstimate, 2)/N);
		
		return temp3/divider;
	}
	
	public double evaluate(Instances test) {
		double classified = 0.0;
		for (int i=0; i< test.size(); i++) {
			if (classify(test.get(i))){
				if (test.get(i).classValue() == classValue) {
					classified++;
				}
			}
		}
		//System.out.println("f:"+unclassified/(double)test.size()+" N:"+test.size());
		return classified/(double)test.size();
	}
}
