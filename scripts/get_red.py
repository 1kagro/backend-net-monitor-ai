# Calcular la red para la nueva dirección IP con su máscara de subred y verificar el nuevo gateway
from ipaddress import IPv4Address, IPv4Network


# ip_address_another = "172.28.31.65"
# subnet_mask_another = "255.255.255.192"
# gateway_another = "172.28.31.125"

def get_network(ip_address_another, subnet_mask_another, gateway_another):
    # Crear el objeto de red para la nueva configuración
    network_another = IPv4Network(f"{ip_address_another}/{subnet_mask_another}", strict=False)
    print(network_another.netmask)
    # Calcular la red y verificar si el nuevo gateway está dentro de esta red
    gateway_address_another = IPv4Address(gateway_another)
    is_gateway_valid_another = gateway_address_another in network_another

    network_info_another = {
        "Red": str(network_another.network_address),
        "Broadcast": str(network_another.broadcast_address),
        "Rango de IPs": f"{str(list(network_another.hosts())[0])} - {str(list(network_another.hosts())[-1])}",
        "Gateway válido": is_gateway_valid_another,
        "Gateway": f"{gateway_another}",
        "Máscara de subred": f"{network_another.netmask}",
    }

    return network_info_another

ip_address_another, subnet_mask_another, gateway_another = input("Ingrese la dirección IP: ej. ip netmask gw").split(" ")

network_info_another = get_network(
    ip_address_another, subnet_mask_another, gateway_another)

network_info_another