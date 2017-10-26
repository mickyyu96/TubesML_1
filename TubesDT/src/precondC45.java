import java.io.Serializable;
import java.util.Map.Entry;

import weka.core.Attribute;

public class precondC45 implements Serializable {
	public Attribute attrprecond;
	public Double valueprecond;
	public Double splitprecond;
	
	public precondC45(Attribute attr, Double val, Double split){
		attrprecond = attr;
		valueprecond = val;
		splitprecond = split;
	}
	
	public precondC45(precondC45 pre){
		attrprecond = pre.attrprecond;
		valueprecond = pre.valueprecond;
		splitprecond = pre.splitprecond;
	}
}
