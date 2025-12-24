# Enterprise Upgrade Guide

> **Powered by HSMN** • Built by Verso Industries

The HighNoon Language Framework Lite edition includes the full HSMN architecture but with scale limits enforced through tamper-proof compiled binaries. The Enterprise edition removes all limits and adds domain-specific modules.

---

## Feature Comparison

| Feature | Lite | Enterprise |
|---------|------|------------|
| **Architecture** | Full HSMN | Full HSMN |
| **Max Parameters** | 20B | Unlimited |
| **Max Reasoning Blocks** | 24 | Unlimited |
| **Max MoE Experts** | 12 | Unlimited |
| **Max Context Length** | 5M tokens | Unlimited |
| **Chemistry Modules** | ❌ | ✅ |
| **Physics Modules** | ❌ | ✅ |
| **Inverse Design** | ❌ | ✅ |
| **Source Access** | Python only | Full source |
| **White-label** | ❌ | ✅ |
| **Support** | Community | 24/7 SLA |

---

## When to Upgrade

Consider upgrading to Enterprise if you need:

1. **Larger Models**: Train or deploy models exceeding 20B parameters
2. **Extended Context**: Process documents longer than 5M tokens
3. **Domain Modules**: Access chemistry, physics, or inverse design capabilities
4. **Custom Binaries**: Modify or recompile the native operations
5. **Dedicated Support**: SLA-backed enterprise support

---

## Upgrade Process

### 1. Contact Sales

Reach out to our sales team to discuss your requirements:

- **Email**: sales@versoindustries.com
- **Web**: https://versoindustries.com/enterprise

### 2. Receive License Key

After completing the licensing agreement, you'll receive:

- Enterprise license key
- Access to enterprise binaries
- Enterprise documentation package

### 3. Install Enterprise Binaries

Replace the Lite binaries with Enterprise versions:

```bash
# Backup existing binaries
cp -r highnoon/_native/bin highnoon/_native/bin.lite

# Install enterprise binaries (from provided package)
tar -xzf highnoon-enterprise-binaries.tar.gz -C highnoon/_native/
```

### 4. Activate License

Activate your enterprise license in code:

```python
import highnoon

# Activate enterprise features
highnoon.activate_enterprise("YOUR-LICENSE-KEY")

# Verify activation
print(highnoon.edition())  # Should print: "enterprise"

# Now you can use unlimited scale
from highnoon.enterprise import FullHSMN

model = FullHSMN(
    total_params=70_000_000_000,  # 70B parameters
    context_length=10_000_000,    # 10M tokens
    num_reasoning_blocks=48,       # 48 blocks
    num_moe_experts=64,           # 64 experts
    enable_chemistry=True,
    enable_physics=True,
)
```

---

## Enterprise-Only Features

### Domain Modules

Enterprise edition includes specialized modules for:

- **Chemistry**: Molecular property prediction, reaction optimization
- **Physics**: Physical simulation integration, constraint satisfaction
- **Inverse Design**: Generative design with property-based optimization

### Extended Configuration

```python
from highnoon.enterprise import EnterpriseConfig

config = EnterpriseConfig(
    # No scale limits
    max_params=None,
    max_context=None,

    # Domain modules
    enable_chemistry=True,
    enable_physics=True,
    enable_inverse_design=True,

    # Advanced features
    enable_distributed_training=True,
    enable_model_parallelism=True,
)
```

---

## Pricing

Enterprise licensing is offered on an annual basis. Contact sales for a custom quote based on your usage requirements:

| Tier | Description |
|------|-------------|
| **Startup** | Up to 5 developers, limited GPU hours |
| **Business** | Up to 25 developers, priority support |
| **Enterprise** | Unlimited developers, dedicated support |
| **Custom** | Tailored to your organization's needs |

---

## Support

### Lite Edition (Community)
- GitHub Issues
- Community forums at [versoindustries.com/messages](https://versoindustries.com/messages)

### Enterprise Edition
- Dedicated support channel
- 24/7 SLA with guaranteed response times
- Onboarding assistance
- Training sessions available

---

## See Also

- [Getting Started](getting-started.md)
- [Distributed Training](distributed_training.md)
- [API Reference](api/)
- [User Guides](guides/)

---

*For more information, visit [versoindustries.com/enterprise](https://versoindustries.com/enterprise)*
