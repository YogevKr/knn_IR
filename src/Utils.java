import java.io.*;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Scanner;

public class Utils {
    public static ArrayList<String> fileToLineList(File file){

        ArrayList<String> list = new ArrayList<String>();
        try {
            Scanner input = new Scanner(file);

            while (input.hasNextLine()) {
                list.add(input.nextLine());
            }

        } catch (IOException e) {
            e.printStackTrace();
        }

        return list;
    }

    public static ArrayList<String> fileToLineList(String filePath){
        return fileToLineList(new File(filePath));
    }


    public static ArrayList<String[]> ReadCsvFile(String filePath) {

        String line = "";
        String cvsSplitBy = ",";

        ArrayList<String[]> out = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {

            while ((line = br.readLine()) != null) {

                out.add(line.split(cvsSplitBy));
            }

        } catch (IOException e) {
            e.printStackTrace();
        }
        return out;
    }

    public static void ListToCSV(File i_File, ArrayList<String[]> i_Data) throws FileNotFoundException {
        PrintWriter pw = new PrintWriter(i_File);
        StringBuilder sb = new StringBuilder();

        Collections.sort(i_Data, new Comparator<String[]>() {
            @Override
            public int compare(final String[] entry1, final String[] entry2) {
                final String time1 = entry1[0];
                final String time2 = entry2[0];
                return time1.compareTo(time2);
            }
        });


        for (String[] doc : i_Data){
            sb.append(doc[0]);
            sb.append(',');
            sb.append(doc[1]);
            sb.append(',');
            sb.append(doc[2]);
            sb.append('\n');
        }

        pw.write(sb.toString());
        pw.close();
    }
}
