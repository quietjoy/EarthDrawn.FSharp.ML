namespace EarthDrawn.FSharp.ML.Source

module LogisticRegression =
    open System
    open MathNet.Numerics.LinearAlgebra
    open MathNet.Numerics.LinearAlgebra.Double
    open System.Drawing
    open System.Windows.Forms
    open System.Windows.Forms.DataVisualization
    open FSharp.Charting

    type LogReg (α: float, λ: float, iterations: int, threshold: float, rawData: Matrix<float>, normalize: Boolean) =
        // features
        member this.features = this.createFeatureMatrix rawData

        // classifications
        member this.classifications = this.createClassificationMatrix rawData

        // list of tuples (int * int) that are the indicies of the 
        // train, cross validation and testing data
        member this.indices = this.getIndicies rawData.RowCount 

        // Build training data
        member this.X_train = this.getSubSetOfMatrix this.features (this.indices.[0])
        member this.y_train = this.getSubSetOfMatrix this.classifications (this.indices.[0])

        // Build cross validation data
        member this.X_cv = this.getSubSetOfMatrix this.features (this.indices.[1])
        member this.y_cv = this.getSubSetOfMatrix this.classifications (this.indices.[1])

        // Build test data
        member this.X_test = this.getSubSetOfMatrix this.features (this.indices.[2])
        member this.y_test = this.getSubSetOfMatrix this.classifications (this.indices.[2])
        
        // initilize theta
        member this.initialTheta = Matrix.Build.Dense(this.X_train.ColumnCount, 1, 0.0)
        
        // Perform Logisitc Regression using gradient descent
        member this.gradients   = this.gradientDescent 0 this.initialTheta
        member this.costs       = this.findCosts this.gradients
        member this.costPlot    = this.generateCostPlot
        member this.finalTheta  = Matrix.Build.DenseOfColumnVectors(this.gradients.Column(this.gradients.ColumnCount-1))
        
        // Use testing data to get error 
        member this.predictions = this.predict this.X_test 
        member this.error       = this.calculateError this.predictions this.y_test

        // sigmoid
        member this.sigmoid (z: Matrix<float>) = 
               z |> Matrix.map (fun x -> 1.0 / (1.0 + (exp -x)))
        
        // normalize a vector to have a mean of 0
        member this.normalize (x: Vector<float>): Vector<float> =
            let max  = x.Maximum()
            let min  = x.Minimum()
            let range = max - min
            let mean = (x |> Vector.sum) / (float x.Count)
            x |> Vector.map (fun x_i -> (x_i-mean)/range)

        // create feature matrix
        member this.createFeatureMatrix (data:Matrix<float>) = 
            if normalize then
                let unNormalizedFeatures = Matrix.Build.Dense(data.RowCount, 1, 1.0)
                                            .Append(data.RemoveColumn(rawData.ColumnCount-1))
                unNormalizedFeatures |> Matrix.mapCols (fun i x -> if (i <> 0) then this.normalize x else x)
            else
                Matrix.Build.Dense(data.RowCount, 1, 1.0)
                                .Append(data.RemoveColumn(rawData.ColumnCount-1))

        // create classification matrix
        member this.createClassificationMatrix (data:Matrix<float>) =
            Matrix.Build.DenseOfColumnVectors(data.Column(data.ColumnCount-1))

        // perform ones step of gradient descent 
        member this.descent (theta:Matrix<float>) =
            let m      = (float this.X_train.RowCount)
            let hx     = this.sigmoid (this.X_train*theta)
            let h      = hx-this.y_train 
            let delt_J = this.X_train 
                            |> Matrix.mapRows (fun i row -> h.[i, 0]*row)
                            |> Matrix.sumCols
                            |> Matrix.Build.DenseOfRowVectors
            let regTerm = theta 
                            |> Matrix.mapi(fun i j y_i -> if (i<>0) then y_i else 0.0) 
                            |> Matrix.sum
            theta - (α*((1.0/m) * delt_J.Transpose()) + (λ/m*regTerm))

        // recursively applies descent function
        member this.gradientDescent (count: int) (gradAccum:Matrix<float>) =
            if count = 0 then
                this.gradientDescent (count+1) (gradAccum.Append(this.descent this.initialTheta)) 
            elif count < iterations then
                let prevTheta = Matrix.Build.DenseOfColumnVectors(gradAccum.Column(count))
                this.gradientDescent (count+1) (gradAccum.Append(this.descent prevTheta))
            else
                gradAccum

        // cost function with feature normalization
        member this.calculateCost (thetaV:Vector<float>): List<float> =
            let m     = (float this.X_train.RowCount) 
            let theta = Matrix.Build.DenseOfColumnVectors(thetaV)
            let hx    = this.sigmoid (this.X_train*theta)
            let sum   = this.y_train
                            |> Matrix.mapi (fun i j y_i -> match y_i with
                                                            | 1.0 -> log(hx.[i, 0])
                                                            | _   -> log(1.0-hx.[i, 0]))
                            |> Matrix.sum
            let regTerm = theta 
                            |> Matrix.mapi(fun i j y_i -> if (i<>0) then (y_i**2.0) else 0.0) 
                            |> Matrix.sum
            [(-1.0/m*sum) + (λ/(2.0*m)*(regTerm))]

        // calculate the cost associated with each gradient
        member this.findCosts (gradients:Matrix<float>): Matrix<float> = 
            let costs = gradients.EnumerateColumns() 
                            |> Seq.map (fun x -> this.calculateCost x)
                            |> Seq.toList
            matrix costs


        // cost plot
        // filter out infinities for now
        // TODO: Make title appear above graph
        member this.generateCostPlot = 
            let costs = this.costs 
                            |> Matrix.toSeq 
                            |> Seq.filter (fun x -> x <> infinity)
            Chart.Line(costs, Name="Cost", Title="Cost Function")
                .WithXAxis(Title="Iteration", Min=0.0, Max=(float iterations))
                .WithYAxis(Title="Cost")


        // Take in the size of the rawData matrix and return a list of tuples that represent
        // 1. indicies of training data (60%) - position 0
        // 2. indicies of c.v. data (20%)     - position 1
        // 3. indicies of testing data (20%)  - position 2
        member this.getIndicies (size:int): (List<(int * int)>) =
            let trainIndex =  int (floor ((float size)*0.6))
            let cvIndex = trainIndex + (int (floor ((float size)*0.2)))
            [(0, trainIndex); (trainIndex, cvIndex); (cvIndex, size)]

        // Generate subset of matrix
        member this.getSubSetOfMatrix (data: Matrix<float>) (indicies: int * int): (Matrix<float>) =
            let rowCount = (snd indicies) - (fst indicies)
            data.SubMatrix((fst indicies), rowCount, 0, data.ColumnCount)

        // Use trained thetas to perform classification
        member this.predict (testSet: Matrix<float>): Matrix<float> =
            let htTheta = this.sigmoid(testSet*this.finalTheta)
            htTheta |> Matrix.map (fun x -> match x with 
                                            | x when x >= threshold -> 1.0
                                            | _ -> 0.0) 
        
        // Calculate error by comparing predictions and actual values
        member this.calculateError (predictions:Matrix<float>) (y: Matrix<float>): float =
            let compare = predictions-y
            let incorrentPredictions = compare 
                                        |> Matrix.toSeq 
                                        |> Seq.filter (fun x -> x <> 0.0)
                                        |> Seq.toList
            ((float incorrentPredictions.Length) / (float y.RowCount))

        // precision
        member this.precision: float =
            let diffSeq  = (this.predictions - this.y_test) |> Matrix.toSeq 
            let truePos  = float ((diffSeq |> Seq.filter (fun elem -> elem = 0.0) |> Seq.toList).Length)
            let falsePos = float ((diffSeq |> Seq.filter (fun elem -> elem = 1.0) |> Seq.toList).Length)
            (truePos / (truePos + falsePos))

        // recall
        member this.recall: float = 
            let diffSeq  = (this.predictions - this.y_test) |> Matrix.toSeq 
            let truePos  = float ((diffSeq |> Seq.filter (fun elem -> elem = 0.0) |> Seq.toList).Length)
            let falseNeg = float ((diffSeq |> Seq.filter (fun elem -> elem = -1.0) |> Seq.toList).Length)
            (truePos / (truePos + falseNeg))

        // F-score
        member this.fScore: float = 
            2.0*(this.precision*this.recall/(this.precision + this.recall)) 

