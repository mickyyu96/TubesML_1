import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;

import java.lang.*;
import java.util.*;
import java.util.Map.Entry;
import java.util.AbstractMap.SimpleEntry;
import java.io.*;

public class ruleC45 implements Serializable {
	private ArrayList<precondC45> precond = new ArrayList<>();
	
	private double classValue;
	private double accuracy = 0;
	private double zErrorEstimate = 1.65;
	
	public ruleC45() {}
	public ruleC45(ruleC45 rule) {
		classValue = rule.classValue;
		accuracy = rule.accuracy;
	} 
	
	public void addPrecond(Attribute attr, double indexAttr, double split) {
		precondC45 newprecond = new precondC45(attr, indexAttr, split);
		precond.add(newprecond);
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
		Double i = 0.0;
		for (precondC45 rule: precond) {
			if (rule.attrprecond.isNumeric()) {
				System.out.println("<"+rule.attrprecond+" = "+rule.splitprecond+"=="+rule.valueprecond+">");
			}else {
				System.out.println("<"+rule.attrprecond+" = "+rule.valueprecond+">");
			}
			i++;
		}
		System.out.println("class value: "+classValue);
		System.out.println("accuracy: "+accuracy);
	}
	
	public void prune(Instances test) {
		int maxacc_key = -1;
		boolean pruned = true;
		accuracy = evaluate(test);
		if (precond.size()>1) {
			//printRule();
			while (pruned) {
				ArrayList<precondC45> lastprecond = new ArrayList<precondC45>(precond);
				for (int i = 0; i < precond.size(); i++) {
					//System.out.println("-------------------removeeeeeee");
					//printRule();
					precond.remove(i);
					//printRule();
					double evalprecond = evaluate(test);
					//System.out.println("evalprecond:"+evalprecond+"> accuracy:"+accuracy);
					if (evalprecond > accuracy) {
						accuracy = evalprecond;
						maxacc_key = i;
					}
					precond = new ArrayList<precondC45>(lastprecond);
					i++;
				}
				
				if (maxacc_key != -1) {
					precond.remove(maxacc_key);
					//System.out.println("-----pruned!----");
					//printRule();
					maxacc_key = -1;
				} else {
					pruned = false;
				}
			}
		}
	}
	
	public boolean classify(Instance data) {
		boolean valid = true;
		Double i = 0.0;
		//printRule();
		for (precondC45 rule: precond) {
			//System.out.println(data);
			//System.out.println(rule.getKey());
			//System.out.println("i: "+i);
			if (rule.attrprecond.isNumeric()) {
				if (rule.valueprecond == 0) {
					
					if ((double) data.value(rule.attrprecond) > rule.splitprecond) {
						valid = false;
					}
				} else if (rule.valueprecond == 1) {
					if ((double) data.value(rule.attrprecond) <= rule.splitprecond) {
						valid = false;
					}
				}
			}else {
				if (data.value(rule.attrprecond) != rule.valueprecond) {
					valid = false;
				}
			}
			i++;
			//System.out.println("->"+i);
		}

		return valid;
	}
	
	public double evaluate(Instances test) {
		double right = 0.0;
		double N = 0.0;
		
		for (int i=0; i< test.size(); i++) {
			if (classify(test.get(i))){
				N++;
				if (test.get(i).classValue() == classValue) {
					right++;
				}
			}
		}
		if (N == 0) {
			//System.out.println("test Nan");
			return 0;
		}
		//System.out.println("right/N="+(right/N));
		return right/N;
	}
}
