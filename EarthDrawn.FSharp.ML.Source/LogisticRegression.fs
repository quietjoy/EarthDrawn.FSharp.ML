namespace EarthDrawn.FSharp.ML.Source

module LogisticRegression =
    open System
    open FSharp.Data
    open MathNet.Numerics.LinearAlgebra
    open MathNet.Numerics.LinearAlgebra.Double

    
    // ********************************    
    // BUILDING MATRICIES AND READING DATA
    // ********************************
    // Turn the array of string[] to an array of float[]
    let castToFloatList (x : string []): List<float> = 
        x |> Seq.map (fun s -> float s) |> Seq.toList

    // Read in csv and return a Matrix
    let readData (path : string): Matrix<float> = 
        let csv = CsvFile.Load(path)
        let f2d = csv.Rows 
                        |> Seq.map (fun x -> x.Columns) 
                        |> Seq.map (fun x -> castToFloatList x)
                        |> Seq.toList
        matrix f2d

    type LogReg (path: string, a0: float, l0: float, it: int) =
        // From constructor
        member this.alpha = a0
        member this.lambda = l0
        member this.iterations = it
        
        // generate raw data from csv whose path is passed in 
        member this.rawData = readData path

        // Build training data - PICK UP HERE
        member this.X_train = this.generateFeatureMatrix this.rawData
        member this.y_train = this.generateClassificationMatrix this.rawData

        // Build cross validation data
        member this.X_cv = 1
        member this.y_cv = 1

        // Build test data
        member this.X_test = 1
        member this.y_test = 1
        
        member this.initialTheta = Matrix.Build.Dense(this.X_train.ColumnCount, 1, 0.0)
        member this.m = (float this.X_train.RowCount)
        
        // Perform Logisitc Regression using gradient descent
        member this.gradients  = this.gradientDescent 0 this.initialTheta
        member this.costs      = this.findCosts this.gradients
        member this.finalTheta = Matrix.Build.DenseOfColumnVectors(this.gradients.Column(this.gradients.ColumnCount-1))

        // GRADIENT DESCENT

        // sigmoid
        member this.sigmoid (z: Matrix<float>) = 
               z |> Matrix.map (fun x -> 1.0 / (1.0 + (exp -x)))

        // Perform ones step of gradient descent 
        member this.descent (theta:Matrix<float>) =
            let hx     = this.sigmoid (this.X_train*theta)
            let h      = hx-this.y_train 
            let delt_J = this.X_train 
                            |> Matrix.mapRows (fun i row -> h.[i, 0]*row)
                            |> Matrix.sumCols
                            |> Matrix.Build.DenseOfRowVectors
            theta - ((1.0/this.m) * (this.alpha * delt_J.Transpose()))

        // Recursively applies descent function
        member this.gradientDescent (count: int) (gradAccum:Matrix<float>) =
            if count = 0 then
                this.gradientDescent (count+1) (gradAccum.Append(this.descent this.initialTheta)) 
            elif count < this.iterations then
                let prevTheta = Matrix.Build.DenseOfColumnVectors(gradAccum.Column(count-1))
                this.gradientDescent (count+1) (gradAccum.Append(this.descent prevTheta))
            else
                gradAccum


        // COST FUNCTION
        // Calculate cost associated with weights
        // Not regularized yet
        member this.calculateCost (thetaV:Vector<float>): float = 
            let theta = Matrix.Build.DenseOfColumnVectors(thetaV)
            let hx    = this.sigmoid (this.X_train*theta)
            let costs = this.y_train 
                            |> Matrix.mapi (fun i j y_i -> match y_i with
                                                            | 1.0 -> hx.[i, 0]
                                                            | _ -> hx.[i, 0])
                            |> Matrix.sum
            -1.0/this.m*costs


        // Given an array of gradients, calculates the cost associated with each gradient
        member this.findCosts (gradients:Matrix<float>) = 
            gradients.EnumerateColumns() 
                        |> Seq.map(fun x -> this.calculateCost x)
                        |> Seq.map (fun x -> [| x |])
                        |> Seq.toArray

        
        // Use trained thetas to perform classification
        member this.predict (testSet: Matrix<float>) =
            let htTheta = this.sigmoid(testSet*this.finalTheta)
            htTheta.[0, 0] >= 0.5
        
        // Generate training data 
        member this.generateFeatureMatrix (data: Matrix<float>): (Matrix<float>) = 
            Matrix.Build
                    .Dense(data.RowCount, 1, 1.0)
                    .Append(data.RemoveColumn(data.ColumnCount-1))

        member this.generateClassificationMatrix (data: Matrix<float>): (Matrix<float>) = 
            Matrix.Build
                    .DenseOfColumnVectors(data.Column(data.ColumnCount-1))
        
            
        // Generating train, cross validation and testing data
//        member this.generateShuffleMatrix (x:Matrix<float>) = 
//            let rng = new Random()
//            let shuffle (mat : 'a Matrix) =
//                let array = Matrix.Build.DenseOfMatrix mat
//                let n = array.RowCount
//                for x in 1..n do
//                    let i = n-x
//                    let j = rng.Next(i+1)
//                    let tmp = array.[i]
//                    array.[i] <- array.[j]
//                    array.[j] <- tmp
//                array
//            shuffle ()