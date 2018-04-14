model= load_model ('my_model1.h5')

score = model.evaluate(X_test, y_test, verbose=0)

print('[INFO] Test loss:', score[0])
print('[INFO] Test accuracy:', score[1])

