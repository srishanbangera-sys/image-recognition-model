def build_model(num_classes):
    model.add(Dense(128, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation = 'softmax'))

    model.compile(
        optimizer = 'adam',
        loss = "sparse_categorical_coressentropy",
        metric = ['accuracy']
    )

    return model