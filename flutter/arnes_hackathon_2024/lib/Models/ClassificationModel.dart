
class ClassificationModel {
  final String name;
  final List<String> versions;
  final String description;


  const ClassificationModel({
    required this.name,
    required this.versions,
    required this.description,
  });

  factory ClassificationModel.fromMap(Map<String, dynamic> map) {
    return ClassificationModel(name: map['name'], versions: [map['version']], description: map['description']);
  }
}
