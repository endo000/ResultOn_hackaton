import 'package:flutter/material.dart';

class TrainView extends StatefulWidget {
  const TrainView({super.key, required this.title});

  final String title;

  @override
  State<TrainView> createState() => _TrainViewState();
}

class _TrainViewState extends State<TrainView> {

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        title: Text(widget.title),
      ),
      body: Stack(
        children: [
          Positioned(
            bottom: 10,
            left: 10,
            right: 10,
            child: TextFormField()
          )
        ],
      ),
    );
  }
}