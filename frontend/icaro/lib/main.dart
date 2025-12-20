import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:firebase_core/firebase_core.dart';
import 'package:firebase_messaging/firebase_messaging.dart';
import 'package:http/http.dart' as http;
import 'package:permission_handler/permission_handler.dart';

import 'firebase_options.dart';

@pragma('vm:entry-point')
Future<void> _firebaseMessagingBackgroundHandler(RemoteMessage message) async {
  await Firebase.initializeApp(options: DefaultFirebaseOptions.currentPlatform);
}

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  await Firebase.initializeApp(
    options: DefaultFirebaseOptions.currentPlatform,
  );

  FirebaseMessaging.onBackgroundMessage(_firebaseMessagingBackgroundHandler);

  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Fall Alerts',
      theme: ThemeData(useMaterial3: true),
      home: const AlertsPage(),
    );
  }
}

class AlertItem {
  final String id;
  final String title;
  final String message;
  final bool alert;

  AlertItem({
    required this.id,
    required this.title,
    required this.message,
    required this.alert,
  });

  factory AlertItem.fromJson(String id, Map<String, dynamic> json) {
    return AlertItem(
      id: id,
      title: (json['title'] ?? '').toString(),
      message: (json['message'] ?? '').toString(),
      alert: (json['alert'] == true),
    );
  }
}

class AlertsPage extends StatefulWidget {
  const AlertsPage({super.key});

  @override
  State<AlertsPage> createState() => _AlertsPageState();
}

class _AlertsPageState extends State<AlertsPage> {
  static const String endpoint = "http://192.168.1.15:5000/api/v1/allerts";

  bool _loading = true;
  String? _error;
  List<AlertItem> _items = [];

  @override
  void initState() {
    super.initState();
    _initNotifications();
    _load();
  }

  Future<void> _initNotifications() async {
    final status = await Permission.notification.status;
    if (!status.isGranted) {
      await Permission.notification.request();
    }

    // Subscribe to topic "fall"
    await FirebaseMessaging.instance.subscribeToTopic("fall");

    // (Optional) handle notifications while app is open
    FirebaseMessaging.onMessage.listen((RemoteMessage message) {
      final title = message.notification?.title ?? "Notification";
      final body = message.notification?.body ?? "";

      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text("$title: $body")),
      );
    });
  }

  Future<void> _load() async {
    setState(() {
      _loading = true;
      _error = null;
    });

    try {
      final res = await http.get(Uri.parse(endpoint));
      if (res.statusCode < 200 || res.statusCode >= 300) {
        throw Exception("HTTP ${res.statusCode}: ${res.body}");
      }

      final decoded = jsonDecode(res.body);

      if (decoded is! Map<String, dynamic>) {
        throw Exception("Unexpected JSON format (expected object/map).");
      }


      final entries = decoded.entries.toList();

      entries.sort((a, b) {
        final ai = int.tryParse(a.key) ?? 0;
        final bi = int.tryParse(b.key) ?? 0;
        return ai.compareTo(bi);
      });

      final items = <AlertItem>[];
      for (final e in entries) {
        final v = e.value;
        if (v is Map<String, dynamic>) {
          items.add(AlertItem.fromJson(e.key, v));
        }
      }

      setState(() {
        _items = items;
        _loading = false;
      });
    } catch (e) {
      setState(() {
        _error = e.toString();
        _loading = false;
      });
    }
  }

  Future<void> _onRefresh() => _load();

  @override
  Widget build(BuildContext context) {
    final body = _loading
        ? const Center(child: CircularProgressIndicator())
        : _error != null
            ? Center(
                child: Padding(
                  padding: const EdgeInsets.all(16),
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Text(
                        "Error:\n$_error",
                        textAlign: TextAlign.center,
                      ),
                      const SizedBox(height: 12),
                      ElevatedButton(
                        onPressed: _load,
                        child: const Text("Retry"),
                      ),
                    ],
                  ),
                ),
              )
            : _items.isEmpty
                ? const Center(child: Text("No alerts"))
                : ListView.separated(
                    physics: const AlwaysScrollableScrollPhysics(),
                    itemCount: _items.length,
                    separatorBuilder: (_, __) => const Divider(height: 1),
                    itemBuilder: (context, index) {
                      final item = _items[index];
                      final icon = item.alert
                          ? Icons.error_outline
                          : Icons.warning_amber_outlined;

                      return ListTile(
                        leading: Icon(icon),
                        title: Text(item.title.isEmpty ? "(no title)" : item.title),
                        subtitle: Text(item.message),
                      );
                    },
                  );

    return Scaffold(
      appBar: AppBar(
        title: const Text("Fall alerts"),
        actions: [
          IconButton(
            onPressed: _load,
            icon: const Icon(Icons.refresh),
            tooltip: "Refresh",
          ),
        ],
      ),
      body: RefreshIndicator(
        onRefresh: _onRefresh,
        child: body,
      ),
    );
  }
}
