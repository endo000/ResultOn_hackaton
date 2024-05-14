import 'dart:convert';
import 'package:arnes_hackathon_2024/Models/ClassificationModel.dart';
import 'package:http/http.dart' as http;
import 'package:flutter_dotenv/flutter_dotenv.dart';


String serverAddress = "86.58.82.202";

Future<List<ClassificationModel>> fetchClassificationModels() async {
  List<ClassificationModel> classificationModels = <ClassificationModel>[];

  final uri = Uri(
    scheme: dotenv.env['SCHEME'] ?? 'http',
    host: dotenv.env['HOST'] ?? "86.58.82.202",
    port: dotenv.env['PORT'] != null ? int.parse(dotenv.env['PORT']!) : 8080,
    path: '/api/v1/models'
  );
  final response = await http.get(uri);

  Map<String, dynamic> decodedData = jsonDecode(response.body);

  if (decodedData.containsKey('models') && decodedData['models']!.isNotEmpty) {
    for (Map<String, dynamic> model in decodedData['models']!) {
      try {
        int index = classificationModels.indexWhere((element) => element.name == model['name']);
        classificationModels[index].versions.add(model['version']);
      } catch(e) {
        classificationModels.add(ClassificationModel.fromMap(model));
      }
    }

  }

  return classificationModels;
}