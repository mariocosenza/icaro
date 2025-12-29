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
      debugShowCheckedModeBanner: false,
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
  final DateTime timestamp;

  AlertItem({
    required this.id,
    required this.title,
    required this.message,
    required this.alert,
    required this.timestamp,
  });

  static String _parseMongoId(dynamic raw) {
    if (raw == null) return '';

    if (raw is Map<String, dynamic>) {
      final oid = raw[r'$oid'];
      if (oid != null) return oid.toString();
    }

    return raw.toString();
  }

  static DateTime _parseTimestamp(dynamic raw) {
    if (raw == null) return DateTime.fromMillisecondsSinceEpoch(0);

    if (raw is String) {
      final dt = DateTime.tryParse(raw);
      if (dt != null) return dt;
      return DateTime.fromMillisecondsSinceEpoch(0);
    }

    if (raw is int) {
      if (raw.abs() >= 1000000000000) {
        return DateTime.fromMillisecondsSinceEpoch(raw);
      }
      return DateTime.fromMillisecondsSinceEpoch(raw * 1000);
    }

    if (raw is Map<String, dynamic>) {
      final dateVal = raw[r'$date'];
      if (dateVal is String) {
        final dt = DateTime.tryParse(dateVal);
        if (dt != null) return dt;
      }
      if (dateVal is Map<String, dynamic>) {
        final nl = dateVal[r'$numberLong'];
        final ms = int.tryParse(nl?.toString() ?? '');
        if (ms != null) return DateTime.fromMillisecondsSinceEpoch(ms);
      }
    }

    return DateTime.fromMillisecondsSinceEpoch(0);
  }

  factory AlertItem.fromDoc(Map<String, dynamic> json) {
    return AlertItem(
      id: _parseMongoId(json['_id']),
      title: (json['title'] ?? '').toString(),
      message: (json['message'] ?? '').toString(),
      alert: (json['alert'] == true),
      timestamp: _parseTimestamp(json['timestamp']),
    );
  }

  // Backward compatibility with your previous { "0": {...} } format
  factory AlertItem.fromKeyedJson(String id, Map<String, dynamic> json) {
    return AlertItem(
      id: id,
      title: (json['title'] ?? '').toString(),
      message: (json['message'] ?? '').toString(),
      alert: (json['alert'] == true),
      timestamp: _parseTimestamp(json['timestamp']),
    );
  }
}

class AlertsPage extends StatefulWidget {
  const AlertsPage({super.key});

  @override
  State<AlertsPage> createState() => _AlertsPageState();
}

class _AlertsPageState extends State<AlertsPage> {
  static const String endpoint = "http://192.168.1.15:8000/api/v1/alerts";

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

    await FirebaseMessaging.instance.subscribeToTopic("fall");

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
      final items = <AlertItem>[];

      if (decoded is Map<String, dynamic>) {
        final rawAlerts = decoded['alerts'];

        // Preferred server response: {"alerts":[{...},{...}]}
        if (rawAlerts is List) {
          for (final el in rawAlerts) {
            if (el is Map<String, dynamic>) {
              items.add(AlertItem.fromDoc(el));
            }
          }
          // Sort by latest timestamp first
          items.sort((a, b) => b.timestamp.compareTo(a.timestamp));
        } else {
          // Legacy keyed map response: {"0": {...}, "1": {...}}
          final entries = decoded.entries.toList();
          entries.sort((a, b) {
            final ai = int.tryParse(a.key) ?? 0;
            final bi = int.tryParse(b.key) ?? 0;
            return ai.compareTo(bi);
          });

          for (final e in entries) {
            final v = e.value;
            if (v is Map<String, dynamic>) {
              items.add(AlertItem.fromKeyedJson(e.key, v));
            }
          }
          // Sort by latest timestamp first
          items.sort((a, b) => b.timestamp.compareTo(a.timestamp));
        }
      }
      // Legacy format: [ {...}, {...} ]
      else if (decoded is List) {
        for (final el in decoded) {
          if (el is Map<String, dynamic>) {
            items.add(AlertItem.fromDoc(el));
          }
        }
        // Sort by latest timestamp first
        items.sort((a, b) => b.timestamp.compareTo(a.timestamp));
      } else {
        throw Exception("Unexpected JSON format (expected map or list).");
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
                        title: Text(
                          item.title.isEmpty ? "(no title)" : item.title,
                        ),
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
