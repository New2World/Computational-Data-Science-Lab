package adaptive;

import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.ArrayList;

public class Tools {
	static NumberFormat formatter = new DecimalFormat("#0.00");
	//System.out.println(formatter.format(4.0));
	public static void printlistln(ArrayList<Integer> list)
	{
		for(int i=0;i<list.size();i++)
		{
			System.out.print(list.get(i)+" ");
		}
		System.out.println();
	}

	public static void printdoublelistln(ArrayList<Double> list)
	{
		for(int i=0;i<list.size();i++)
		{
			if(i>0 && (list.get(i)-list.get(i-1))<0.01)
			{
				System.out.print(formatter.format(list.get(i))+" ");
				System.out.println();
				return;
			}
			else
			{
				System.out.print(formatter.format(list.get(i))+" ");
			}

		}
		System.out.println();
	}

	public static void printdoublelistln(ArrayList<Double> list, int size)
	{
		if(size>list.size())
		{
			throw new ArithmeticException("printdoublelistln size>list.size()");

		}
		for(int i=0;i<size;i++)
		{

			if(i<list.size())
			{
				System.out.print(formatter.format(list.get(i))+" ");
			}


		}
		System.out.println();
	}

	public static void printintlistln(ArrayList<Integer> list, int size)
	{
		if(size>list.size())
		{
			throw new ArithmeticException("printintlistln size>list.size()");

		}
		for(int i=0;i<size;i++)
		{

			if(i<list.size())
			{
				System.out.print(formatter.format(list.get(i))+" ");
			}


		}
		System.out.println();
	}

    public static void printElapsedTime(long startTime, long endTime){
        long execTime = endTime - startTime;
        System.out.printf(">>> elapsed time: %02d:%02d:%02d.%03d\n", execTime/3600000, execTime/60000%60, execTime/1000%60, execTime%1000);
    }

    public static void printElapsedTime(long startTime, long endTime, String name){
        long execTime = endTime - startTime;
        System.out.printf(">>> %s elapsed time: %02d:%02d:%02d.%03d\n", name, execTime/3600000, execTime/60000%60, execTime/1000%60, execTime%1000);
    }
}