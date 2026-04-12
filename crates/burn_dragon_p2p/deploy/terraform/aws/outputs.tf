output "edge_url" {
  description = "Public browser edge URL."
  value       = "https://${var.edge_domain_name}"
}

output "bootstrap_instance_id" {
  description = "EC2 instance id for the bootstrap edge."
  value       = aws_instance.bootstrap.id
}

output "bootstrap_public_ip" {
  description = "Elastic IP attached to the bootstrap edge."
  value       = aws_eip.bootstrap.public_ip
}

output "seed_node_tcp_multiaddr" {
  description = "TCP bootstrap multiaddr advertised to native peers."
  value       = "/dns4/${var.edge_domain_name}/tcp/${var.p2p_port}"
}

output "seed_node_quic_multiaddr" {
  description = "QUIC bootstrap multiaddr advertised to native peers."
  value       = "/dns4/${var.edge_domain_name}/udp/${var.p2p_port}/quic-v1"
}

output "secret_parameter_prefix" {
  description = "SSM parameter prefix read by the bootstrap edge at runtime."
  value       = var.secret_parameter_prefix
}
