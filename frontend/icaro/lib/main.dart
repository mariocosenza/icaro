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
  static const String _defaultBackendHost = "192.168.1.15:8000";
  String _backendHost = _defaultBackendHost;

  bool _loading = true;
  String? _error;
  List<AlertItem> _items = [];

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _bootstrap();
    });
  }

  Future<void> _bootstrap() async {
    await _initNotifications();
    await _promptForBackend();
    await _load();
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
        SnackBar(
          behavior: SnackBarBehavior.floating,
          backgroundColor: Colors.grey.shade900,
          content: Row(
            children: [
              const Icon(Icons.notifications, color: Colors.white),
              const SizedBox(width: 8),
              Expanded(
                child: Text(
                  "$title: $body",
                  style: const TextStyle(color: Colors.white),
                ),
              ),
            ],
          ),
        ),
      );
    });
  }

  String _normalizeBackendHost(String input) {
    final trimmed = input.trim();
    if (trimmed.isEmpty) return _defaultBackendHost;
    if (trimmed.startsWith("http://") || trimmed.startsWith("https://")) {
      return trimmed;
    }
    return trimmed;
  }

  String get _endpoint {
    final normalized = _normalizeBackendHost(_backendHost);
    final base = normalized.startsWith("http://") || normalized.startsWith("https://")
        ? normalized
        : "http://$normalized";
    final trimmedBase = base.endsWith("/") ? base.substring(0, base.length - 1) : base;
    return "$trimmedBase/api/v1/alerts";
  }

  Future<void> _promptForBackend() async {
    final controller = TextEditingController(text: _backendHost);
    final result = await showDialog<String>(
      context: context,
      barrierDismissible: false,
      builder: (context) {
        return AlertDialog(
          title: const Text("Backend IP"),
          content: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const Text(
                "Enter the backend IP (optionally with port).",
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 12),
              TextField(
                controller: controller,
                keyboardType: TextInputType.url,
                decoration: const InputDecoration(
                  hintText: "192.168.1.15:8000",
                  border: OutlineInputBorder(),
                ),
              ),
            ],
          ),
          actions: [
            TextButton(
              onPressed: () {
                Navigator.of(context).pop(_defaultBackendHost);
              },
              child: const Text("Use default"),
            ),
            ElevatedButton(
              onPressed: () {
                Navigator.of(context).pop(controller.text);
              },
              child: const Text("Connect"),
            ),
          ],
        );
      },
    );

    if (!mounted) return;
    if (result != null && result.trim().isNotEmpty) {
      setState(() {
        _backendHost = result.trim();
      });
    }
  }

  Future<void> _load() async {
    setState(() {
      _loading = true;
      _error = null;
    });

    try {
      final res = await http.get(Uri.parse(_endpoint));
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

  String _formatTimestamp(DateTime ts) {
    if (ts.millisecondsSinceEpoch == 0) return "Unknown time";
    final local = ts.toLocal();
    String two(int v) => v.toString().padLeft(2, '0');
    return "${local.year}-${two(local.month)}-${two(local.day)} "
        "${two(local.hour)}:${two(local.minute)}:${two(local.second)}";
  }

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
                      final color = item.alert ? Colors.red : Colors.orange;

                      return Padding(
                        padding: const EdgeInsets.symmetric(
                          horizontal: 12,
                          vertical: 6,
                        ),
                        child: Card(
                          elevation: 0,
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(12),
                            side: BorderSide(color: color.withOpacity(0.35)),
                          ),
                          child: ListTile(
                            leading: Container(
                              width: 42,
                              height: 42,
                              decoration: BoxDecoration(
                                color: color.withOpacity(0.15),
                                shape: BoxShape.circle,
                              ),
                              child: Icon(icon, color: color),
                            ),
                            title: Text(
                              item.title.isEmpty ? "(no title)" : item.title,
                              style: const TextStyle(
                                fontWeight: FontWeight.w600,
                              ),
                            ),
                            subtitle: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Text(item.message),
                                const SizedBox(height: 4),
                                Text(
                                  _formatTimestamp(item.timestamp),
                                  style: Theme.of(context).textTheme.bodySmall,
                                ),
                              ],
                            ),
                          ),
                        ),
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
