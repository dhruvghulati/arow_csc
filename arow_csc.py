import arow
import sys

if __name__ == "__main__":

    trainDataLines = open(sys.argv[1]).readlines()
    # test.dat
    testDataLines = open(sys.argv[2]).readlines()
    # open_cost_1.predict
    predictFile = sys.argv[3]

    train_data = [arow.train_instance_from_svm_input(line) for line in trainDataLines]
    test_data = [arow.test_instance_from_svm_input(line) for line in testDataLines]
    cl = arow.AROW()
    # print [cl.predict(d).label for d in test_data]
    # print [d.costs for d in test_data]

    cl.train(train_data)

    predictions = [cl.predict(d, verbose=True).label for d in test_data]

    # print predictions
    # print [cl.predict(d, verbose=True).featureValueWeights for d in test_data]
    # print [d.costs for d in test_data]

    f = open(predictFile, 'w')

    for i, prediction in enumerate(predictions):
        line = str(int(prediction)) + " " + str(i)
        f.write(line+"\n")

    f.close()