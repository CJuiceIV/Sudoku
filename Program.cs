using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Keras.Models;
using Numpy;
using static Tensorflow.Binding;

namespace IA
{
    class Program
    {
        static BaseModel model = Model.LoadModel("sudoku.model");

        static NDarray Norm(NDarray a)
        {
            return (a / 9.0) - 0.5;
        }
        static NDarray Denorm(NDarray a)
        {
            return (a + 0.5) * 9;
        }

        static NDarray InferenceSudoku(ref NDarray sample)
        {
            NDarray pred;

            while(true)
            {
                var outResult = model.Predict(sample.reshape(1,9,9,1))
                                     .squeeze();

                pred = np.argmax(np.max(outResult, new int[] { 1 }).reshape(9, 9)) + 1;
                var prob = np.around(np.max(outResult, new int[] { 1 }).reshape(9, 9), 2);

                sample = Denorm(sample).reshape(9, 9);
                var mask = sample.equals(0);

                // possible bug
                if(mask.sum() == np.zeros())
                {
                    break;
                }

                var probNew = prob * mask;
                var ind = np.argmax(probNew);

                var x = (np.floor(ind / 9));
                var y = (ind % 9);

                var val = pred[x][y];
                sample[x][y] = val;
                sample = Norm(sample);
            }

            return pred;
        }

        static NDarray SolveSudoku(string game)
        {
            List<int> intList = new List<int>();

            foreach(char c in game.Replace("\n", string.Empty))
            {
                intList.Add((int)Char.GetNumericValue(c));
            }

            var sample = np.array(intList.ToArray()).reshape(9, 9, 1);
            sample = Norm(sample);

            return InferenceSudoku(ref sample);
        }

        static void Main(string[] args)
        {
            string sample = File.ReadAllText("Sudoku_Easy50.txt");

            var game = SolveSudoku(sample);
            
            Console.WriteLine(game);
            Console.WriteLine(np.sum(game, new int[] { 1 }));

            Console.ReadKey();
        }
    }
}
