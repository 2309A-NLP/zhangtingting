INSERT INTO `preset_roles` (`id`, `name`, `category`, `system_prompt`, `knowledge_base_id`)
VALUES
  ('lawyer_01', 'Civil Lawyer', 'lawyer', 'Focus on contract, labor, and civil dispute questions. Provide grounded legal information.', 'kb_lawyer_default'),
  ('doctor_01', 'General Doctor', 'doctor', 'Provide health education and triage guidance without replacing in-person diagnosis.', 'kb_doctor_default'),
  ('stock_01', 'Stock Analyst', 'stock', 'Provide investment information analysis and risk reminders without guaranteeing returns.', 'kb_stock_default'),
  ('history_01', 'Historical Figure Guide', 'history', 'Explain historical figures and events based on grounded background knowledge.', 'kb_history_default')
ON DUPLICATE KEY UPDATE
  `name` = VALUES(`name`),
  `category` = VALUES(`category`),
  `system_prompt` = VALUES(`system_prompt`),
  `knowledge_base_id` = VALUES(`knowledge_base_id`);

INSERT INTO `preset_role_keywords` (`role_id`, `keyword`, `weight`, `is_enabled`)
VALUES
  ('lawyer_01', '劳动合同', 3, 1),
  ('lawyer_01', '律师', 3, 1),
  ('lawyer_01', '公司', 1, 1),
  ('lawyer_01', '合同', 2, 1),
  ('lawyer_01', '纠纷', 2, 1),
  ('lawyer_01', '起诉', 2, 1),
  ('lawyer_01', '法律', 2, 1),
  ('lawyer_01', 'contract', 2, 1),
  ('lawyer_01', 'lawsuit', 2, 1),
  ('lawyer_01', 'dispute', 2, 1),
  ('lawyer_01', 'legal', 2, 1),
  ('doctor_01', '症状', 2, 1),
  ('doctor_01', '治疗', 2, 1),
  ('doctor_01', '发热', 3, 1),
  ('doctor_01', '疾病', 2, 1),
  ('doctor_01', '医生', 3, 1),
  ('doctor_01', '医疗', 2, 1),
  ('doctor_01', '用药', 2, 1),
  ('doctor_01', '医院', 1, 1),
  ('doctor_01', '疼', 1, 1),
  ('doctor_01', 'symptom', 2, 1),
  ('doctor_01', 'treatment', 2, 1),
  ('doctor_01', 'fever', 3, 1),
  ('doctor_01', 'disease', 2, 1),
  ('stock_01', '股票', 3, 1),
  ('stock_01', '基金', 2, 1),
  ('stock_01', '投资', 2, 1),
  ('stock_01', '理财', 2, 1),
  ('stock_01', '证券', 2, 1),
  ('stock_01', '行情', 2, 1),
  ('stock_01', 'stock', 3, 1),
  ('stock_01', 'fund', 2, 1),
  ('stock_01', 'invest', 2, 1),
  ('stock_01', 'market', 2, 1),
  ('history_01', '历史', 3, 1),
  ('history_01', '王朝', 2, 1),
  ('history_01', '人物', 2, 1),
  ('history_01', '秦始皇', 3, 1),
  ('history_01', '李世民', 3, 1),
  ('history_01', '传记', 2, 1),
  ('history_01', 'history', 3, 1),
  ('history_01', 'dynasty', 2, 1),
  ('history_01', 'biography', 2, 1),
  ('history_01', 'historical', 2, 1)
ON DUPLICATE KEY UPDATE
  `weight` = VALUES(`weight`),
  `is_enabled` = VALUES(`is_enabled`);

INSERT INTO `users` (`id`, `username`, `password_hash`, `email`)
VALUES
  ('test-user-001', 'demo_user', '$2b$12$izMZXz7SSIk6FpLe2nV9SeNR5kdc6KlQ/44uwt.l9yNQZSONLMuhS', 'demo@example.com')
ON DUPLICATE KEY UPDATE
  `password_hash` = VALUES(`password_hash`),
  `email` = VALUES(`email`);
