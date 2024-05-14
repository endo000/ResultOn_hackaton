import 'package:arnes_hackathon_2024/API/api.dart';
import 'package:arnes_hackathon_2024/Views/TrainView.dart';
import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import '../Models/ClassificationModel.dart';
import 'ClassificationView.dart';

class ModelsView extends StatefulWidget {
  const ModelsView({super.key, required this.title});

  final String title;

  @override
  State<ModelsView> createState() => _ModelsViewState();
}

class _ModelsViewState extends State<ModelsView> {
  Future<List<ClassificationModel>> classificationModels = fetchClassificationModels();
  // List<ClassificationModel> classificationModels = <ClassificationModel>[
  //   const ClassificationModel(name: "test1", versions: ["1"], description: "test1"),
  //   const ClassificationModel(name: "test1", versions: ["1"], description: "test1"),
  //   const ClassificationModel(name: "test1", versions: ["1"], description: "test1"),
  //   const ClassificationModel(name: "test1", versions: ["1", "2", "3"], description: "test1")
  // ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        title: Text(widget.title),
      ),
      floatingActionButton: FloatingActionButton(
        child: const Icon(Icons.add),
        onPressed: () {
          Navigator.push(
            context,
            CupertinoPageRoute(builder: (context) => const TrainView(title: 'Train')),
          );
        },
      ),
      floatingActionButtonLocation: FloatingActionButtonLocation.endFloat,
      body: FutureBuilder<List<ClassificationModel>>(
          future: fetchClassificationModels(),
          builder: (BuildContext context, AsyncSnapshot<List<ClassificationModel>> snapshot) {
            if (snapshot.hasData) {
              return ListView.builder(
                  padding: const EdgeInsets.all(8),
                  itemCount: snapshot.data?.length,
                  itemBuilder: (BuildContext context, int index) {
                    List<Widget> modelButtons = <Widget>[];
                    for (String version in snapshot.data![index].versions) {
                      modelButtons.add(
                          OutlinedButton(
                              onPressed: () {
                                Navigator.push(
                                  context,
                                  CupertinoPageRoute(builder: (context) =>
                                      ClassificationView(
                                          title: 'Image Classification',
                                          model: snapshot.data![index],
                                          version: version,
                                      )),
                                );
                              },
                              child: Text(version)
                          )
                      );
                    }
                    return ExpansionTile(
                      title: Text(snapshot.data![index].name),
                      children: [
                        Text(snapshot.data![index].description),
                        const Text("Choose a version:"),
                        ...modelButtons
                      ],
                    );
                  }
              );
            } else {
              return Container();
            }
          }
      )
    );
  }
}