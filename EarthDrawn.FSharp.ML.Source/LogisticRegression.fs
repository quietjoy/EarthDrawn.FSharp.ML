namespace EarthDrawn.FSharp.ML.Source

module LogisticRegression =
    open System
    open MathNet.Numerics.LinearAlgebra
    open MathNet.Numerics.LinearAlgebra.Double

    type LogReg (α: float, λ: float, iterations: int, rawData: Matrix<float>) =
        // all features
        member this.features = Matrix.Build
                                .Dense(rawData.RowCount, 1, 1.0)
                                .Append(rawData.RemoveColumn(rawData.ColumnCount-1))

        // all classifications
        member this.classifications = Matrix.Build
                                        .DenseOfColumnVectors(rawData.Column(rawData.ColumnCount-1))

        // get a list of tuples (int * int) that represent the indicies of the 
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
        member this.finalTheta  = Matrix.Build.DenseOfColumnVectors(this.gradients.Column(this.gradients.ColumnCount-1))
        
        // Use testing data to get error 
        member this.predictions = this.predict this.X_test 
        member this.error       = this.calculateError this.predictions this.y_test

        // sigmoid
        member this.sigmoid (z: Matrix<float>) = 
               z |> Matrix.map (fun x -> 1.0 / (1.0 + (exp -x)))
        
        // GRADIENT DESCENT
        // Perform ones step of gradient descent 
        member this.descent (theta:Matrix<float>) =
            let m      = (float this.X_train.RowCount)
            let hx     = this.sigmoid (this.X_train*theta)
            let h      = hx-this.y_train 
            let delt_J = this.X_train 
                            |> Matrix.mapRows (fun i row -> h.[i, 0]*row)
                            |> Matrix.sumCols
                            |> Matrix.Build.DenseOfRowVectors
            theta - ((1.0/m) * (α * delt_J.Transpose()))

        // Recursively applies descent function
        member this.gradientDescent (count: int) (gradAccum:Matrix<float>) =
            if count = 0 then
                this.gradientDescent (count+1) (gradAccum.Append(this.descent this.initialTheta)) 
            elif count < iterations then
                let prevTheta = Matrix.Build.DenseOfColumnVectors(gradAccum.Column(count-1))
                this.gradientDescent (count+1) (gradAccum.Append(this.descent prevTheta))
            else
                gradAccum


        // COST FUNCTION
        // Calculate cost associated with weights
        // Not regularized yet
        member this.calculateCost (thetaV:Vector<float>): List<float> =
            let m     = (float this.X_train.RowCount) 
            let theta = Matrix.Build.DenseOfColumnVectors(thetaV)
            let hx    = this.sigmoid (this.X_train*theta)
            let costs = this.y_train 
                            |> Matrix.mapi (fun i j y_i -> match y_i with
                                                            | 1.0 -> hx.[i, 0]
                                                            | _ -> hx.[i, 0])
                            |> Matrix.sum
            [(-1.0/m*costs)]

        // Given an array of gradients, calculates the cost associated with each gradient
        member this.findCosts (gradients:Matrix<float>) = 
            let costs = gradients.EnumerateColumns() 
                            |> Seq.map (fun x -> this.calculateCost x)
                            |> Seq.toList
            matrix costs

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
        member this.predict (testSet: Matrix<float>) =
            let htTheta = this.sigmoid(testSet*this.finalTheta)
            htTheta |> Matrix.map (fun x -> match x with 
                                            | x when x >= 0.5 -> 1.0
                                            | _ -> 0.0) 
        
        member this.calculateError (predictions:Matrix<float>) (y: Matrix<float>): float =
            let compare = predictions-y
            let incorrentPredictions = compare 
                                        |> Matrix.toSeq 
                                        |> Seq.filter (fun x -> x <> 0.0)
                                        |> Seq.toList
            float (incorrentPredictions.Length / y.RowCount)